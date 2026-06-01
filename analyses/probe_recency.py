"""Probe: does L19's residual encode *recent prompt features* rather than an internal value?

If the L19 → Q_max readability (R²=0.84) is mediated by the activation reflecting
which letter was recently chosen / rewarded (rather than the model running an internal
Q-tracking computation), then probes for *letter identity* of recency-derived targets
should decode at near-ceiling accuracy. The "value representation" is then a
correlational shadow of recency, not a mechanism.

Three targets are decoded from each layer's decision-token residual:

  last_letter      (20-way) — token id of letter chosen at trial t-1
                              (pure recency: model just emitted this token)
  best_Q_letter    (20-way) — token id of letter at argmax cogmod Q at trial t
                              ("the best arm to pick" per the fitted cogmod)
  Q_max_scalar     (regress)— reference; should reproduce locate.py R²≈0.84 at L19

Multinomial logistic regression (ridge-regularized, L-BFGS) for the 20-class targets;
ridge regression for the scalar target. Evaluation: top-1 accuracy unrestricted
(over all 20 letters) and *episode-restricted* (mask logits for letters not in the
current episode's 4-letter set). Random-chance episode-restricted accuracy = 25%.

The diagnostic:

  best_Q_letter and last_letter both near-ceiling
    AND their predictions agree on most trials
    → L19's "value" signal is recency reading, not internal computation.

    python -m analyses.probe_recency \
        --activations artifacts/activations/llama8b_residual.pt \
        --q           artifacts/value/bestrl_Q.pt \
        --dataset     artifacts/datasets/llama8b_dataset.json \
        --layers 10 15 19 25 28 31 \
        --out         artifacts/locate/llama8b_recency_probe.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

EPS = 1e-9


def fit_logreg(X_tr: torch.Tensor, y_tr: torch.Tensor, n_classes: int,
               l2: float = 1e-2, max_iter: int = 200) -> nn.Linear:
    """Multinomial logistic regression via L-BFGS on cross-entropy + L2."""
    head = nn.Linear(X_tr.shape[1], n_classes).to(X_tr.device)
    opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=max_iter,
                            tolerance_grad=1e-7, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = head(X_tr)
        loss = F.cross_entropy(logits, y_tr) + l2 * (head.weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return head


def fit_ridge(X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    p = X.shape[1]
    XtX = X.T @ X + alpha * torch.eye(p, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(XtX, X.T @ y)


def r2(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor) -> float:
    pred = X @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum().clamp(min=EPS)
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--q",           type=pathlib.Path, required=True)
    parser.add_argument("--dataset",     type=pathlib.Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[10, 15, 19, 25, 28, 31])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--l2",    type=float, default=1e-2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    # ---- load activations / Q / dataset ----
    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]                  # [n_eps_local, T, n_layers, 4096] fp16
    ep_ids = act_data["episode_ids"]
    n_eps_local, T, n_layers, hidden_dim = activations.shape

    q_data = torch.load(args.q, weights_only=False)
    Q = q_data["Q"][ep_ids].float()                        # [n_eps_local, T, n_arms]
    actions = q_data["actions"][ep_ids]                    # [n_eps_local, T]  long

    logs = json.loads(args.dataset.read_text())
    ep_ids_int = [int(e) for e in ep_ids]
    letters_per_ep = [logs[ep]["letters"] for ep in ep_ids_int]  # list of 4-letter lists

    # ---- universal letter token id map (20 classes) ----
    tok = AutoTokenizer.from_pretrained(args.model)
    letter_pool: dict[str, int] = {}
    for lp in letters_per_ep:
        for L in lp:
            if L not in letter_pool:
                ids = tok(L, add_special_tokens=False)["input_ids"]
                assert len(ids) == 1, f"letter {L!r} not single token"
                letter_pool[L] = ids[0]
    sorted_letters = sorted(letter_pool, key=lambda L: letter_pool[L])
    letter_to_class = {L: i for i, L in enumerate(sorted_letters)}
    n_classes = len(sorted_letters)
    print(f"letter pool ({n_classes} letters): {sorted_letters}")
    print(f"token ids:                       {[letter_pool[L] for L in sorted_letters]}")

    # ---- build per-trial targets ----
    # last_letter[ep, t]   = letter chosen at t-1   (undefined at t=0)
    # best_Q_letter[ep, t] = letter at argmax Q[t]  (undefined at t=0 — Q[0] uniform)
    # Q_max_scalar[ep, t]  = max Q[t]
    last_letter_cls   = torch.full((n_eps_local, T), -1, dtype=torch.long)
    best_Q_letter_cls = torch.full((n_eps_local, T), -1, dtype=torch.long)
    Q_max_scalar      = torch.zeros((n_eps_local, T), dtype=torch.float32)
    # Track which 4 letter-classes are valid per (ep, t) for episode-restricted eval
    valid_mask = torch.zeros((n_eps_local, T, n_classes), dtype=torch.bool)

    for e, letters in enumerate(letters_per_ep):
        ep_class_ids = [letter_to_class[L] for L in letters]   # 4 class ids for this ep
        for t in range(T):
            valid_mask[e, t, ep_class_ids] = True
            if t >= 1:
                last_letter_cls[e, t] = letter_to_class[letters[int(actions[e, t - 1])]]
            q_argmax = int(Q[e, t].argmax())
            best_Q_letter_cls[e, t] = letter_to_class[letters[q_argmax]]
            Q_max_scalar[e, t] = float(Q[e, t].max())

    # Skip t=0 (last_letter undefined; Q[0] is uniform → degenerate targets)
    activations = activations[:, 1:]
    last_letter_cls   = last_letter_cls[:, 1:]
    best_Q_letter_cls = best_Q_letter_cls[:, 1:]
    Q_max_scalar      = Q_max_scalar[:, 1:]
    valid_mask        = valid_mask[:, 1:]
    T_pred = T - 1

    # ---- 80/10/10 by episode (same convention as locate.py) ----
    n_train = int(round(0.8 * n_eps_local))
    n_val   = int(round(0.1 * n_eps_local))
    sl_tr = slice(0, n_train)
    sl_te = slice(n_train + n_val, n_eps_local)
    n_test = n_eps_local - n_train - n_val
    print(f"split: {n_train} train / {n_val} val / {n_test} test episodes "
          f"× {T_pred} trials each")

    # ---- agreement diagnostic on TEST: how often does last_letter == best_Q_letter? ----
    agree_te = (last_letter_cls[sl_te] == best_Q_letter_cls[sl_te]).float().mean()
    print(f"\nTest-set base rate: last_letter == best_Q_letter on "
          f"{float(agree_te) * 100:.1f}% of trials   (perseveration baseline)")

    # ---- per-layer probes ----
    results: list[dict] = []
    t0 = time.time()
    for layer in args.layers:
        A = activations[:, :, layer, :].float().to(args.device)            # [n_eps, T_pred, 4096]
        X_tr_raw = A[sl_tr].reshape(-1, hidden_dim)
        X_te_raw = A[sl_te].reshape(-1, hidden_dim)
        mu = X_tr_raw.mean(dim=0, keepdim=True)
        sd = X_tr_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
        X_tr = (X_tr_raw - mu) / sd
        X_te = (X_te_raw - mu) / sd

        valid_te = valid_mask[sl_te].reshape(-1, n_classes).to(args.device)  # [N_te, 20]
        layer_result = {"layer": layer}

        # --- two multiclass targets ---
        for target_name, target_tensor in [
            ("last_letter",   last_letter_cls),
            ("best_Q_letter", best_Q_letter_cls),
        ]:
            y_tr = target_tensor[sl_tr].reshape(-1).to(args.device)
            y_te = target_tensor[sl_te].reshape(-1).to(args.device)

            head = fit_logreg(X_tr, y_tr, n_classes, l2=args.l2)
            with torch.no_grad():
                logits_te = head(X_te)                                       # [N_te, 20]
                # Unrestricted (over all 20 letters)
                pred_unr = logits_te.argmax(dim=-1)
                acc_unr = float((pred_unr == y_te).float().mean())
                # Episode-restricted (mask non-episode letters)
                logits_masked = logits_te.masked_fill(~valid_te, float("-inf"))
                pred_res = logits_masked.argmax(dim=-1)
                acc_res = float((pred_res == y_te).float().mean())

            layer_result[target_name] = {
                "acc_unrestricted_20way": acc_unr,
                "acc_episode_restricted_4way": acc_res,
                "chance_unrestricted": 1.0 / n_classes,
                "chance_restricted":   1.0 / 4,
            }
            print(f"  layer {layer:2d}  {target_name:<14} "
                  f"acc(20-way)={acc_unr:.3f}  acc(4-way)={acc_res:.3f}")

        # --- scalar reference: Q_max via ridge ---
        y_tr_s = Q_max_scalar[sl_tr].reshape(-1).to(args.device)
        y_te_s = Q_max_scalar[sl_te].reshape(-1).to(args.device)
        # standardise target for numerical parity
        ym, ys = y_tr_s.mean(), y_tr_s.std().clamp(min=EPS)
        y_tr_n = (y_tr_s - ym) / ys
        y_te_n = (y_te_s - ym) / ys
        best = None
        for alpha in [1.0, 10.0, 100.0, 1_000.0, 10_000.0]:
            beta = fit_ridge(X_tr, y_tr_n.unsqueeze(-1), alpha).squeeze(-1)
            r2_te = r2(X_te, y_te_n, beta)
            if best is None or r2_te > best["r2"]:
                best = {"alpha": alpha, "r2": r2_te}
        layer_result["Q_max_scalar"] = {
            "r2_test": best["r2"], "best_alpha": best["alpha"],
        }
        print(f"  layer {layer:2d}  Q_max_scalar     R²(test)={best['r2']:+.3f}  "
              f"(α={best['alpha']:.0f})    [{time.time()-t0:.0f}s]")

        # --- agreement between probe predictions (only meaningful when both > chance) ---
        with torch.no_grad():
            # Re-fit briefly to capture both predictions in one pass
            head_last = fit_logreg(X_tr, last_letter_cls[sl_tr].reshape(-1).to(args.device),
                                   n_classes, l2=args.l2, max_iter=80)
            head_bestQ = fit_logreg(X_tr, best_Q_letter_cls[sl_tr].reshape(-1).to(args.device),
                                    n_classes, l2=args.l2, max_iter=80)
            pred_last  = head_last(X_te).masked_fill(~valid_te, float("-inf")).argmax(dim=-1)
            pred_bestQ = head_bestQ(X_te).masked_fill(~valid_te, float("-inf")).argmax(dim=-1)
            agree_preds = float((pred_last == pred_bestQ).float().mean())
        layer_result["probe_pred_agreement"] = agree_preds
        print(f"  layer {layer:2d}  probe predictions agree on {agree_preds*100:.1f}% of "
              f"test trials\n")

        results.append(layer_result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model_id":         act_data["model_id"],
        "value_model":      q_data["model_name"],
        "letter_pool":      sorted_letters,
        "letter_token_ids": [letter_pool[L] for L in sorted_letters],
        "n_eps_train":      n_train,
        "n_eps_test":       n_test,
        "n_trials_per_ep":  T_pred,
        "test_base_rate_last_eq_bestQ": float(agree_te),
        "layers":           results,
    }, indent=2))
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
