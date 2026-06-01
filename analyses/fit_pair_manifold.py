"""Fit between-arms manifolds M_h and M_y for the V2 steering experiment (fixed letters).

The bandit analog of the paper's weekday addition: for an arm pair (i, j) the 1-D concept
is the *relative value* r = Q[j] − Q[i], running from strongly-i (r < 0) through balanced
(r ≈ 0, the model uncertain between i and j) to strongly-j (r > 0). We restrict to trials
where {i, j} are the top-2 arms, so the transition is a clean i↔j hand-off uncontaminated
by a third arm. Because letters are FIXED across episodes, arm index == letter throughout,
so centroids are letter-coherent (the lesson from the bin-centroid failure).

For each unordered pair (i < j):
  M_h: PCA-64 centroids binned by r → one cubic spline per PCA coord → manifold_grid.
  M_y: mean 4-arm choice distribution per r-bin → Hellinger embed → tangent-plane spline →
       sphere_grid / dist_grid (the natural behavior curve from arm i to arm j).

PCA is fit ONCE over all trials at the layer (paper-style global subspace) and shared.

Output (`artifacts/manifolds/<name>.pt`): shared {pca_basis, pca_mean, pca_var_explained,
layer, ...} plus pairs[f"{i}_{j}"] = {centroids_pca, manifold_grid, r_grid, r_centers,
r_counts, dist_centroids, dist_grid, sphere_grid, base_point, tangent_centroids, i, j, n}.

    python -m analyses.fit_pair_manifold \
        --activations artifacts/activations/llama8b_fixed_residual.pt \
        --q artifacts/value/bestrl_Q_fixed.pt \
        --dataset artifacts/datasets/llama8b_fixed.json \
        --layer 19 --n-bins 25 \
        --out artifacts/manifolds/llama8b_fixed_pairs_L19.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline

from analyses.fit_behavior_manifold import exp_map, log_map


def _quantile_edges(x, n_bins):
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    return edges


def _bin_centroids(coord, values, edges):
    """Group `values` [N, D] by which bin of `edges` `coord` [N] falls in.
    Returns centroids [n_used, D], centers [n_used], counts [n_used]."""
    n_bins = len(edges) - 1
    idx = np.clip(np.searchsorted(edges[1:-1], coord, side="right"), 0, n_bins - 1)
    cents, centers, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c > 0:
            cents.append(values[m].mean(axis=0))
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            counts.append(c)
    return np.stack(cents), np.array(centers), np.array(counts)


def _fit_splines(x, Y, w, s, n_grid):
    """One UnivariateSpline per column of Y [n, D] over x [n]; sample on a grid."""
    splines = [UnivariateSpline(x, Y[:, d], w=w, s=s, k=3) for d in range(Y.shape[1])]
    grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    sampled = np.stack([spl(grid) for spl in splines], axis=1)
    return grid, sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--trial-min", type=int, default=1)
    parser.add_argument("--trial-max", type=int, default=0, help="restrict trials to [min,max); 0=all")
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--smoothing", type=float, default=None)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]                       # [n_eps, T, n_layers, hidden]
    ep_ids = act_data["episode_ids"]
    n_eps, T, n_layers, hidden = activations.shape

    q_data = torch.load(args.q, weights_only=False)
    Q = q_data["Q"][ep_ids].cpu().numpy()                       # [n_eps, T, n_arms]
    n_arms = Q.shape[2]

    logs = json.loads(args.dataset.read_text())
    logs = [logs[e] for e in ep_ids]
    choice = np.array([l["choice_dists"] for l in logs], dtype=np.float32)[..., :n_arms]
    choice = choice / choice.sum(axis=-1, keepdims=True)        # drop 'other', renormalize
    letters = logs[0]["letters"]

    # ---- global PCA over the chosen trial window at the layer (skip trial 0) ----
    tmin = max(1, args.trial_min); tmax = args.trial_max if args.trial_max > 0 else T
    A = activations[:, tmin:tmax, args.layer, :].float().reshape(-1, hidden)   # [N, hidden]
    A_mean = A.mean(dim=0)
    _, S, Vh = torch.linalg.svd(A - A_mean, full_matrices=False)
    pca_basis = Vh[: args.pca_dim]
    var_exp = (S ** 2) / (S ** 2).sum()
    A_pca_all = ((A - A_mean) @ pca_basis.T).cpu().numpy()              # [N, pca_dim]
    print(f"layer {args.layer}: PCA-{args.pca_dim} cum var {float(var_exp[:args.pca_dim].sum()):.3f}; "
          f"letters={letters}")

    # per-trial views aligned with A_pca_all rows (eps × (T-1))
    Qp = Q[:, tmin:tmax].reshape(-1, n_arms)                            # [N, n_arms]
    Cp = choice[:, tmin:tmax].reshape(-1, n_arms)                       # [N, n_arms]
    order = np.argsort(-Qp, axis=1)                                     # top-2 = order[:, :2]

    pairs = {}
    for i, j in itertools.combinations(range(n_arms), 2):
        sel = (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)
        n = int(sel.sum())
        if n < args.n_bins * 3:
            print(f"  pair ({i},{j}) [{letters[i]}-{letters[j]}]: only {n} top-2 trials — skipped")
            continue

        r = (Qp[sel, j] - Qp[sel, i]).astype(np.float64)               # intrinsic coord
        A_pair = A_pca_all[sel]                                         # [n, pca_dim]
        C_pair = Cp[sel]                                               # [n, n_arms]
        edges = _quantile_edges(r, args.n_bins)

        cen_pca, r_centers, counts = _bin_centroids(r, A_pair, edges)
        cen_dist, _, _ = _bin_centroids(r, C_pair, edges)
        n_used = len(r_centers)
        w = np.sqrt(counts.astype(np.float64))
        s = float(args.smoothing) if args.smoothing is not None else float(n_used)

        r_grid, manifold_grid = _fit_splines(r_centers, cen_pca, w, s, args.n_grid)

        # M_y: Hellinger embed → tangent → spline → sphere
        sqrt_c = np.sqrt(cen_dist)
        b_star = sqrt_c.mean(axis=0)
        b_star = b_star / np.linalg.norm(b_star)
        tangents = log_map(b_star, sqrt_c)
        _, tangent_grid = _fit_splines(r_centers, tangents, w, s, args.n_grid)
        sphere_grid = exp_map(b_star, tangent_grid)
        dist_grid = sphere_grid ** 2
        dist_grid = dist_grid / dist_grid.sum(axis=1, keepdims=True)

        pairs[f"{i}_{j}"] = {
            "i": i, "j": j, "letters": [letters[i], letters[j]], "n": n,
            "centroids_pca": torch.tensor(cen_pca, dtype=torch.float32),
            "manifold_grid": torch.tensor(manifold_grid, dtype=torch.float32),
            "r_grid":        torch.tensor(r_grid, dtype=torch.float32),
            "r_centers":     torch.tensor(r_centers, dtype=torch.float32),
            "r_counts":      torch.tensor(counts, dtype=torch.long),
            "dist_centroids": torch.tensor(cen_dist, dtype=torch.float32),
            "dist_grid":     torch.tensor(dist_grid, dtype=torch.float32),
            "sphere_grid":   torch.tensor(sphere_grid, dtype=torch.float32),
            "base_point":    torch.tensor(b_star, dtype=torch.float32),
            "tangent_centroids": torch.tensor(tangents, dtype=torch.float32),
        }
        # endpoints: P(i) at r_min end, P(j) at r_max end
        print(f"  pair ({i},{j}) [{letters[i]}-{letters[j]}]: n={n}, {n_used} bins; "
              f"P(i)→ {dist_grid[0, i]:.2f}@r_min, P(j)→ {dist_grid[-1, j]:.2f}@r_max")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "pairs":            pairs,
        "pca_basis":        pca_basis.cpu(),
        "pca_mean":         A_mean.cpu(),
        "pca_var_explained": var_exp[: args.pca_dim].cpu(),
        "layer":            args.layer,
        "letters":          letters,
        "n_arms":           n_arms,
        "n_bins":           args.n_bins,
        "n_grid":           args.n_grid,
        "model_id":         act_data["model_id"],
        "value_model":      q_data["model_name"],
    }, args.out)
    print(f"\nsaved → {args.out}  ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
