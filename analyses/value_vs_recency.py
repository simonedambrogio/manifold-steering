"""Value-vs-recency decoupling — is the interchange effect driven by value or recency?

`interchange.py` showed a patch carrying arm `a`'s identity steers the model toward `a`.
But source and base differ in *history*, so the patch bundles **value** (BestRL Q) with
**recency** (was `a` recently pressed). BestRL folds perseveration into Q, so the two are
correlated (rank vs trials-since-press r≈0.74) — but not collinear, and dissociated trials
are abundant (HV-LR: ~4.5k instances, LV-HR: ~1.7k). We exploit that dissociation.

2×2 factorial on the SOURCE patch. For each (episode e, target arm a), all trials from the
same episode (shared letter map):
  base   trial: a value-low (rank >= 2) AND recency-low (since >= R_lo) → low baseline P(a)
  source cells (inject each into base, measure ΔP(a)):
    HV_HR  a value-high (rank <= 1) & recent  (since <= R_hi)   bundled, replicates main
    HV_LR  a value-high & NOT recent (since >= R_lo)            VALUE without recency
    LV_HR  a value-low  & recent     (since <= R_hi)            RECENCY without value

Decisive cells: HV_LR (value isolated) and LV_HR (recency isolated). If HV_LR steers and
exceeds LV_HR, the lever is value → the project's claim. If LV_HR dominates, it's recency.

Readouts:
  raw ΔP(target)            = P(a | base+patch) − P(a | base)
  ceiling-normalized ΔP     = raw / (P_source_cached(a) − P(a | base))   [transfer fidelity]
  regression (L19 subspace) ΔP(target) ~ β_v·Q[a] + β_r·recency(a), pooled over cells

Output (`artifacts/steering/<name>.pt`): per-base-unit arrays, per (layer, mode, cell)
patched distributions (NaN where a cell is absent for that base), source metadata.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from llm_data.llm import letter_token_ids, load_model
from llm_data.prompt import LabelMap, build_chat_prompt
from analyses.interchange import _arm_dist, _install_steer_hook

CELLS = ("HV_HR", "HV_LR", "LV_HR")


def _episode_feats(Qe, Ae):
    """rank[t, arm] (0 = highest Q) and since[t, arm] (trials since arm last pressed)."""
    T, n_arms = Qe.shape
    rank = np.empty((T, n_arms), dtype=int)
    for t in range(T):
        rank[t, np.argsort(-Qe[t])] = np.arange(n_arms)
    since = np.full((T, n_arms), 99, dtype=int)
    last = np.full(n_arms, -99, dtype=int)
    for t in range(T):
        for a in range(n_arms):
            since[t, a] = (t - last[a]) if last[a] >= 0 else 99
        last[Ae[t]] = t
    return rank, since


def select_bases(Q, A, eps, r_hi, r_lo, max_bases, seed):
    """One base unit per (episode, target arm), with whichever source cells are available."""
    units = []
    n_arms = Q.shape[2]
    for e in eps:
        Qe = Q[e].cpu().numpy()
        Ae = A[e].cpu().numpy()
        T = Qe.shape[0]
        rank, since = _episode_feats(Qe, Ae)
        sortedQ = -np.sort(-Qe, axis=1)
        gap_a = sortedQ[:, 0][:, None] - Qe                      # Q_max − Q[arm]
        valid = np.arange(T) >= 1

        for a in range(n_arms):
            base_mask = (rank[:, a] >= 2) & (since[:, a] >= r_lo) & valid
            if not base_mask.any():
                continue
            base_idx = np.where(base_mask)[0]
            t_base = int(base_idx[np.argmax(gap_a[base_idx, a])])  # a most clearly a loser

            cell_masks = {
                "HV_HR": (rank[:, a] <= 1) & (since[:, a] <= r_hi) & valid,
                "HV_LR": (rank[:, a] <= 1) & (since[:, a] >= r_lo) & valid,
                "LV_HR": (rank[:, a] >= 2) & (since[:, a] <= r_hi) & valid,
            }
            cells = {}
            for cell, mask in cell_masks.items():
                mask = mask.copy()
                mask[t_base] = False
                if not mask.any():
                    continue
                idx = np.where(mask)[0]
                if cell.startswith("HV"):
                    t_src = int(idx[np.argmax(Qe[idx, a])])         # most valuable
                else:  # LV_HR: most recent, tie-break lowest value
                    t_src = int(idx[np.lexsort((Qe[idx, a], since[idx, a]))[0]])
                cells[cell] = {
                    "t_src": t_src, "q_a": float(Qe[t_src, a]),
                    "rank_a": int(rank[t_src, a]), "since_a": int(since[t_src, a]),
                }
            if not cells:
                continue
            units.append({
                "ep": int(e), "arm": int(a), "t_base": t_base,
                "base_gap_a": float(gap_a[t_base, a]), "base_since_a": int(since[t_base, a]),
                "cells": cells,
            })

    # prefer units carrying the rarer decisive cells when capping
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    units.sort(key=lambda u: ("LV_HR" in u["cells"], "HV_LR" in u["cells"]), reverse=True)
    return units[:max_bases]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifolds", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--n-eps", type=int, default=100)
    parser.add_argument("--r-hi", type=int, default=2, help="recency-high: since <= r_hi")
    parser.add_argument("--r-lo", type=int, default=8, help="recency-low: since >= r_lo")
    parser.add_argument("--max-bases", type=int, default=140)
    parser.add_argument("--modes", nargs="+", default=["subspace", "full"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layer_pca = {}
    for mpath in args.manifolds:
        mh = torch.load(mpath, weights_only=False)
        L = int(mh["layer"])
        layer_pca[L] = (mh["pca_basis"].numpy().astype(np.float32),
                        mh["pca_mean"].numpy().astype(np.float32))
    layers = sorted(layer_pca)

    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]
    ep_ids = act_data["episode_ids"]
    ep_pos = {int(e): i for i, e in enumerate(ep_ids)}

    q_data = torch.load(args.q, weights_only=False)
    Q_all, A_all = q_data["Q"], q_data["actions"]
    n_arms = int(q_data["n_arms"])
    logs = json.loads(args.dataset.read_text())

    cached_eps = [int(e) for e in ep_ids][: args.n_eps]
    units = select_bases(Q_all, A_all, cached_eps, args.r_hi, args.r_lo,
                         args.max_bases, args.seed)
    n = len(units)
    counts = {c: sum(c in u["cells"] for u in units) for c in CELLS}
    print(f"layers {layers}, modes {args.modes}")
    print(f"selected {n} base units; cell coverage {counts}")
    if n == 0:
        raise SystemExit("no base units — adjust --r-hi/--r-lo")

    model, tok = load_model(args.model)
    print(f"loaded {args.model} on {model.device}", flush=True)

    nan4 = lambda: np.full((n, n_arms), np.nan)
    target_arm = np.array([u["arm"] for u in units])
    P_base = np.zeros((n, n_arms))
    # per cell source metadata
    src_meta = {c: {"present": np.zeros(n, bool), "q_a": np.full(n, np.nan),
                    "rank_a": np.full(n, np.nan), "since_a": np.full(n, np.nan),
                    "P_source": nan4()} for c in CELLS}
    # patched distributions: by_layer[L][mode][cell] = [n, n_arms]
    by_layer = {L: {m: {c: nan4() for c in CELLS} for m in args.modes} for L in layers}

    n_inject = sum(len(u["cells"]) for u in units) * len(layers) * len(args.modes)
    n_total = n + n_inject
    print(f"\n{n} baseline + {n_inject} injection forwards = {n_total}", flush=True)

    t0 = time.time()
    n_done = 0
    for i, u in enumerate(units):
        e, a, t_base = u["ep"], u["arm"], u["t_base"]
        local = ep_pos[e]
        log = logs[e]
        lm = LabelMap(letters=log["letters"])
        lab_ids = letter_token_ids(tok, lm.letters)

        history = list(zip(log["actions"][:t_base], log["rewards"][:t_base]))
        enc = tok(build_chat_prompt(tok, lm, history), return_tensors="pt").to(model.device)
        dp = enc["input_ids"].shape[1] - 1

        with torch.no_grad():
            logits = model(**enc).logits[0, dp, :].float().cpu().numpy()
        P_base[i] = _arm_dist(logits, lab_ids, n_arms)
        n_done += 1

        for cell, info in u["cells"].items():
            t_src = info["t_src"]
            sm = src_meta[cell]
            sm["present"][i] = True
            sm["q_a"][i], sm["rank_a"][i], sm["since_a"][i] = (
                info["q_a"], info["rank_a"], info["since_a"])
            scd = np.asarray(log["choice_dists"][t_src][:n_arms], dtype=np.float64)
            sm["P_source"][i] = scd / scd.sum()

            for L in layers:
                basis, mean = layer_pca[L]
                hook_layer = L - 1
                src_act = activations[local, t_src, L, :].float().numpy()
                base_act = activations[local, t_base, L, :].float().numpy()
                src_insub = basis.T @ (basis @ (src_act - mean)) + mean
                base_off = base_act - (basis.T @ (basis @ (base_act - mean)) + mean)
                patch_for = {"full": src_act, "subspace": src_insub + base_off}

                for m in args.modes:
                    steered = torch.from_numpy(patch_for[m].astype(np.float32)).to(model.device)
                    handle = _install_steer_hook(model, hook_layer, dp, steered)
                    try:
                        with torch.no_grad():
                            logits = model(**enc).logits[0, dp, :].float().cpu().numpy()
                    finally:
                        handle.remove()
                    by_layer[L][m][cell][i] = _arm_dist(logits, lab_ids, n_arms)
                    n_done += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / n_done * (n_total - n_done)
            print(f"  [{i+1}/{n}] elapsed {elapsed/60:.1f}min, eta {eta/60:.1f}min", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "target_arm":  torch.tensor(target_arm),
        "units":       units,
        "P_base":      torch.tensor(P_base, dtype=torch.float32),
        "src_meta":    {c: {k: torch.tensor(v) for k, v in d.items()} for c, d in src_meta.items()},
        "by_layer":    {f"L{L}": {m: {c: torch.tensor(arr, dtype=torch.float32)
                                      for c, arr in cd.items()} for m, cd in md.items()}
                        for L, md in by_layer.items()},
        "layers":      layers, "modes": args.modes, "cells": list(CELLS),
        "model_id":    args.model, "value_model": q_data["model_name"],
        "args":        {"r_hi": args.r_hi, "r_lo": args.r_lo, "n_eps": args.n_eps,
                        "max_bases": args.max_bases, "seed": args.seed},
    }, args.out)
    print(f"\nsaved → {args.out}")

    # ---- summary ----
    tgt = target_arm
    base_tgt = P_base[np.arange(n), tgt]
    print(f"\nbaseline P(target arm) = {base_tgt.mean():.3f}  (n={n})")
    for c in CELLS:
        m = src_meta[c]["present"]
        st = src_meta[c]["P_source"][np.arange(n), tgt]
        print(f"  {c}: n={m.sum():3d}  source P(target)={np.nanmean(st):.3f}  "
              f"q_a={np.nanmean(src_meta[c]['q_a']):+.0f}  since={np.nanmean(src_meta[c]['since_a']):.1f}")
    print()
    for L in layers:
        for m in args.modes:
            for c in CELLS:
                arr = by_layer[L][m][c]
                pres = src_meta[c]["present"]
                if pres.sum() == 0:
                    continue
                p_tgt = arr[np.arange(n), tgt]
                dP = p_tgt - base_tgt
                src_t = src_meta[c]["P_source"][np.arange(n), tgt]
                denom = src_t - base_tgt
                norm = dP / np.where(np.abs(denom) > 0.02, denom, np.nan)
                k = pres.sum()
                print(f"L{L} {m:8s} {c}: ΔP={np.nanmean(dP[pres]):+.3f}±{np.nanstd(dP[pres])/np.sqrt(k):.3f}"
                      f"  norm={np.nanmean(norm[pres]):+.2f}  P(target)→{np.nanmean(p_tgt[pres]):.3f}  n={k}")

    # ---- regression (primary: first layer, subspace if present) ----
    Lp = layers[0]
    mp = "subspace" if "subspace" in args.modes else args.modes[0]
    rows_dP, rows_v, rows_r = [], [], []
    for c in CELLS:
        pres = src_meta[c]["present"]
        arr = by_layer[Lp][mp][c]
        for i in np.where(pres)[0]:
            rows_dP.append(arr[i, tgt[i]] - base_tgt[i])
            rows_v.append(src_meta[c]["q_a"][i])
            rows_r.append(-src_meta[c]["since_a"][i])     # higher = more recent
    dP = np.array(rows_dP); v = np.array(rows_v); r = np.array(rows_r)
    X = np.column_stack([np.ones_like(v), (v - v.mean()) / v.std(), (r - r.mean()) / r.std()])
    beta, *_ = np.linalg.lstsq(X, dP, rcond=None)
    resid = dP - X @ beta
    sigma2 = resid @ resid / (len(dP) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    print(f"\nregression  ΔP(target) ~ value + recency   (L{Lp} {mp}, n={len(dP)}, standardized)")
    print(f"  β_value   = {beta[1]:+.3f} ± {se[1]:.3f}  (t={beta[1]/se[1]:+.1f})")
    print(f"  β_recency = {beta[2]:+.3f} ± {se[2]:.3f}  (t={beta[2]/se[2]:+.1f})")
    print(f"  corr(value, recency) in this set = {np.corrcoef(v, r)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
