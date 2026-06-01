"""Evaluate pair-steering with BOTH naturalness metrics (post-hoc, no model needed).

The two metrics answer different questions, and they disagree in our non-folding setting:
  E_BC (nearest-point)   "is the induced distribution ON the M_y curve (anywhere)?"
                         — pacing-blind; favored linear in early runs.
  matched-fraction MSE   "does the trajectory reproduce the natural transition in ORDER?"
                         — the manifold mechanism (even steps in r); favors manifold.

For each (pair, base) and mode we compute, over the K waypoints:
  ebc          Σ min-Bhattacharyya(induced 4-arm dist, M_y curve)
  pj_mse       mean (P(j) − M_y P(j) at matched fraction)²
  full_mse     mean ||induced − M_y at matched fraction||²  (4-arm)
  offaxis      mean mass on the non-{i,j} arms
Paired t-tests (manifold vs linear_full) over all (pair, base) samples.

    python -m analyses.eval_pair_steering \
        --pairs artifacts/manifolds/llama8b_fixed_pairs_L19.pt \
        --steering artifacts/steering/llama8b_fixed_pair_steering_L19.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from scipy.stats import ttest_rel


def bc_nearest(p, grid):
    ov = (np.sqrt(p)[None, :] * np.sqrt(grid)).sum(1)
    return -np.log(np.clip(ov, 1e-12, 1.0)).min()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    ap.add_argument("--steering", type=pathlib.Path, required=True)
    ap.add_argument("--linear", default="linear_full")
    args = ap.parse_args()

    mf = torch.load(args.pairs, weights_only=False)
    st = torch.load(args.steering, weights_only=False)
    res, K, na = st["results"], st["K"], st["n_arms"]
    lin = args.linear

    rows = {m: {"ebc": [], "pj_mse": [], "full_mse": [], "offaxis": []}
            for m in ("manifold", lin)}
    print(f"layer {st['layer']}  ({args.steering.name})")
    print(f"{'pair':>7} {'metric':>9} {'manifold':>9} {'linear':>9}")
    for pk, R in res.items():
        i, j = R["i"], R["j"]
        others = [a for a in range(na) if a not in (i, j)]
        dg = mf["pairs"][pk]["dist_grid"].numpy()
        tgt = dg[np.linspace(0, len(dg) - 1, K).round().astype(int)]      # [K, na]
        per = {}
        for m in ("manifold", lin):
            d = R[f"dist_{m}"].numpy()                                     # [nb, K, na+1]
            nb = d.shape[0]
            arms = d[..., :na] / d[..., :na].sum(-1, keepdims=True)        # [nb, K, na]
            ebc = np.array([sum(bc_nearest(arms[b, k], dg) for k in range(K)) for b in range(nb)])
            pj = ((arms[..., j] - tgt[None, :, j]) ** 2).mean(1)           # [nb]
            full = ((arms - tgt[None]) ** 2).mean((1, 2))                  # [nb]
            oa = arms[..., others].sum(-1).mean(1)                         # [nb]
            for key, v in [("ebc", ebc), ("pj_mse", pj), ("full_mse", full), ("offaxis", oa)]:
                rows[m][key].append(v)
            per[m] = (ebc.mean(), pj.mean(), full.mean(), oa.mean())
        print(f"{pk:>7} {'full_mse×1e3':>9}  {per['manifold'][2]*1e3:8.3f} {per[lin][2]*1e3:8.3f}")

    print(f"\n=== aggregate (n={sum(len(x) for x in rows['manifold']['ebc'])} pair×base, "
          f"manifold vs {lin}) ===")
    for key, scale, lab, better in [("ebc", 1, "E_BC (nearest-pt)", "↓"),
                                    ("pj_mse", 1e3, "P(j) MSE ×1e3 (tracking)", "↓"),
                                    ("full_mse", 1e3, "full MSE ×1e3 (tracking)", "↓"),
                                    ("offaxis", 1, "off-axis mass", "↓")]:
        man = np.concatenate(rows["manifold"][key])
        liv = np.concatenate(rows[lin][key])
        t, p = ttest_rel(man, liv)
        win = "manifold" if man.mean() < liv.mean() else "linear"
        print(f"  {lab:28s} manifold={man.mean()*scale:7.3f}  linear={liv.mean()*scale:7.3f}"
              f"  → {win:8s} (t={t:+.1f}, p={p:.1e})")


if __name__ == "__main__":
    main()
