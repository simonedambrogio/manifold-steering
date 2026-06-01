"""Locate value-related signal in Llama's activations (per-layer ridge regression).

For each layer L, fit a ridge regression from the decision-token residual stream a_L
to a set of **episode-invariant scalar targets** derived from BestRL's per-trial Q.
Reports R² per (layer, target) on train/val/test with by-episode 80/10/10 splits;
selects ridge α per layer (joint across targets) by mean val R² from a small grid.

Targets are deliberately scalar and episode-invariant — unlike per-arm Q, which is
confounded by per-episode letter randomization (arm-index has no stable meaning across
episodes; see status.md):

  - Q_chosen              :  value of the arm Llama chose at trial t      (Q[t, a_t])
  - Q_max                 :  highest Q across arms at trial t              (max(Q[t]))
  - Q_spread              :  max(Q[t]) − min(Q[t])  (how discriminable arms are)
  - Q_chosen_minus_mean   :  Q[t, a_t] − mean(Q[t])  (relative value of chosen)

Outputs:
  - <out>.json  — per (layer, target) R² + chosen α per layer (compact)
  - <out>.pt    — beta weights + train standardization stats per layer (for downstream)

    python -m analyses.locate \
        --activations artifacts/activations/llama8b_residual.pt \
        --q artifacts/value/bestrl_Q.pt \
        --out artifacts/locate/llama8b_bestrl_scalars
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Callable

import torch


EPS = 1e-9


def _Q_chosen(Q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Q[t, a_t]. Q: [B, T, n_arms], actions: [B, T] → [B, T]."""
    return Q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _Q_sorted_at(rank: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Closure returning the Q value at the given rank (0 = max, n_arms−1 = min) per trial."""
    def fn(Q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        sorted_q, _ = Q.sort(dim=-1, descending=True)
        return sorted_q[..., rank]
    return fn


# Targets are episode-invariant scalars:
#   Q_chosen  — value of the arm the LLM actually chose (action-dependent baseline)
#   Q_max     — value of the best arm (Q[rank=0])
#   Q_2nd     — value of the 2nd-best arm (Q[rank=1])
#   Q_3rd     — value of the 3rd-best arm (Q[rank=2])
#   Q_min     — value of the worst arm (Q[rank=3])
TARGET_FNS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "Q_chosen": _Q_chosen,
    "Q_max":    _Q_sorted_at(0),
    "Q_2nd":    _Q_sorted_at(1),
    "Q_3rd":    _Q_sorted_at(2),
    "Q_min":    _Q_sorted_at(3),
}


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    p = X.shape[1]
    XtX = X.T @ X + alpha * torch.eye(p, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(XtX, X.T @ Y)


def r2(X: torch.Tensor, Y: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    Y_pred = X @ beta
    ss_res = ((Y - Y_pred) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp(min=EPS)
    return 1.0 - ss_res / ss_tot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="path stem; writes <out>.json and <out>.pt")
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0],
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    # ---- load ----
    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]  # [n_eps, T, n_layers, hidden_dim] fp16
    ep_ids = act_data["episode_ids"]
    n_eps, T, n_layers, hidden_dim = activations.shape

    q_data = torch.load(args.q, weights_only=False)
    Q = q_data["Q"][ep_ids]                # [n_eps, T, n_arms]
    actions = q_data["actions"][ep_ids]    # [n_eps, T]

    target_names = list(TARGET_FNS.keys())
    n_targets = len(target_names)
    targets = torch.stack(
        [TARGET_FNS[n](Q.float(), actions) for n in target_names], dim=-1
    )  # [n_eps, T, n_targets]

    print(f"activations {tuple(activations.shape)}  Q {tuple(Q.shape)}")
    print(f"targets {target_names}:")
    for i, name in enumerate(target_names):
        c = targets[:, :, i]
        print(f"  {name:>22}: mean {float(c.mean()):+.2f}  std {float(c.std()):.2f}  "
              f"range [{float(c.min()):+.1f}, {float(c.max()):+.1f}]")

    # ---- skip trial 0 (uninformative — Q[0] is uniform init, scalar targets degenerate) ----
    activations = activations[:, 1:]
    targets = targets[:, 1:]
    T_pred = T - 1

    # ---- by-episode 80/10/10 split ----
    n_train = int(round(0.8 * n_eps))
    n_val = int(round(0.1 * n_eps))
    sl_train = slice(0, n_train)
    sl_val = slice(n_train, n_train + n_val)
    sl_test = slice(n_train + n_val, n_eps)
    n_test = n_eps - n_train - n_val
    print(f"split: {n_train}/{n_val}/{n_test} eps × {T_pred} predicted trials  "
          f"= {n_train * T_pred}/{n_val * T_pred}/{n_test * T_pred} samples")

    # ---- per-layer ridge ----
    per_layer: list[dict] = []
    betas = torch.empty(n_layers, hidden_dim, n_targets)
    train_means = torch.empty(n_layers, hidden_dim)
    train_stds = torch.empty(n_layers, hidden_dim)
    best_alphas = torch.empty(n_layers)

    t0 = time.time()
    for layer in range(n_layers):
        A = activations[:, :, layer, :].float().to(args.device)
        Y_full = targets.to(args.device)

        X_train = A[sl_train].reshape(-1, hidden_dim)
        X_val = A[sl_val].reshape(-1, hidden_dim)
        X_test = A[sl_test].reshape(-1, hidden_dim)
        Y_train = Y_full[sl_train].reshape(-1, n_targets)
        Y_val = Y_full[sl_val].reshape(-1, n_targets)
        Y_test = Y_full[sl_test].reshape(-1, n_targets)

        mu = X_train.mean(dim=0, keepdim=True)
        sd = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
        X_train = (X_train - mu) / sd
        X_val = (X_val - mu) / sd
        X_test = (X_test - mu) / sd

        best = None
        for alpha in args.alphas:
            beta = fit_ridge(X_train, Y_train, alpha)
            r2_v = r2(X_val, Y_val, beta)
            score = float(r2_v.mean())                    # joint α selection: mean across targets
            if best is None or score > best["score"]:
                best = {
                    "alpha":    alpha,
                    "score":    score,
                    "beta":     beta,
                    "r2_train": r2(X_train, Y_train, beta).cpu().tolist(),
                    "r2_val":   r2_v.cpu().tolist(),
                    "r2_test":  r2(X_test, Y_test, beta).cpu().tolist(),
                }

        per_layer.append({
            "layer":      layer,
            "best_alpha": best["alpha"],
            "r2_train":   dict(zip(target_names, best["r2_train"])),
            "r2_val":     dict(zip(target_names, best["r2_val"])),
            "r2_test":    dict(zip(target_names, best["r2_test"])),
        })
        betas[layer] = best["beta"].cpu()
        train_means[layer] = mu.squeeze(0).cpu()
        train_stds[layer] = sd.squeeze(0).cpu()
        best_alphas[layer] = best["alpha"]

        elapsed = time.time() - t0
        r2_str = "  ".join(f"{n}={r:+.3f}" for n, r in zip(target_names, best["r2_test"]))
        print(f"  layer {layer:2d}: α={best['alpha']:>7.0f}  test R²: {r2_str}  ({elapsed:.0f}s)")

    # ---- save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_out = args.out.with_suffix(".json")
    pt_out = args.out.with_suffix(".pt")

    json_out.write_text(json.dumps({
        "model_id":           act_data["model_id"],
        "value_model":        q_data["model_name"],
        "value_model_params": q_data["model_params"],
        "n_eps_train":        n_train,
        "n_eps_val":          n_val,
        "n_eps_test":         n_test,
        "n_predict_trials":   T_pred,
        "target_names":       target_names,
        "alphas_tried":       args.alphas,
        "layers":             per_layer,
    }, indent=2))

    torch.save({
        "betas":         betas,         # [n_layers, hidden_dim, n_targets]
        "train_means":   train_means,   # [n_layers, hidden_dim]
        "train_stds":    train_stds,    # [n_layers, hidden_dim]
        "best_alphas":   best_alphas,   # [n_layers]
        "layer_index":   torch.arange(n_layers),
        "target_names":  target_names,
    }, pt_out)

    print(f"\nsaved {json_out}  +  {pt_out}")
    for name in target_names:
        scores = [p["r2_test"][name] for p in per_layer]
        best_l = max(range(n_layers), key=lambda i: scores[i])
        print(f"  best layer for {name:>22}: {best_l:2d}  R²={scores[best_l]:+.3f}")


if __name__ == "__main__":
    main()
