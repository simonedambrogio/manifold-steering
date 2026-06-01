"""Fit the 1-D behavior manifold M_y over Llama's per-trial choice distributions.

Recipe:
  1. For each (episode, trial), sort the 4-arm choice distribution by current Q ordering
     (descending: best Q first). This makes the 4-vector episode-invariant despite the
     per-episode arm-letter randomization.
  2. Bin trials by Q_max using the SAME bin edges as M_h (loaded from --manifold-h).
  3. Per bin: average sorted distributions → behavior centroid (point on the simplex Δ³).
  4. Hellinger transform: √p element-wise → centroid on the unit sphere's positive orthant.
  5. Pick a base point b_* (normalised mean of √centroids); log-map all centroids into the
     tangent plane at b_* — a flat Euclidean space tangent to the sphere.
  6. Fit per-coordinate smoothing splines (4 of them — one per arm rank) on the tangent
     vectors, weighted by sqrt(bin_count).
  7. Pre-sample on the same Q_max grid as M_h: evaluate splines → tangent vectors →
     exp-map back to sphere → square to get valid distributions on the simplex.

Output (`artifacts/manifolds/<name>.pt`):
  sphere_grid           [n_grid, n_arms]  — γ_My sampled, on unit sphere
  dist_grid             [n_grid, n_arms]  — corresponding probability distributions
  hellinger_centroids   [n_used_bins, n_arms]  — √centroid_distribution per bin
  dist_centroids        [n_used_bins, n_arms]
  base_point            [n_arms]  — b_* on the sphere (positive orthant)
  tangent_centroids     [n_used_bins, n_arms]  — log-mapped to tangent at b_*
  bin_centers, bin_counts, bin_edges, manifold_q_grid

    python -m analyses.fit_behavior_manifold \
        --dataset artifacts/datasets/llama8b_dataset.json \
        --q artifacts/value/bestrl_Q.pt \
        --manifold-h artifacts/manifolds/llama8b_layer19_Qmax.pt \
        --n-eps 100 \
        --out artifacts/manifolds/llama8b_My_Qmax.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline


def log_map(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sphere log-map at `base`. base [d], points [n, d] → tangent vectors [n, d]."""
    inner = np.clip(points @ base, -1.0, 1.0)                          # [n]
    theta = np.arccos(inner)                                            # [n]
    factor = np.where(theta < 1e-10, 1.0, theta / np.sin(np.clip(theta, 1e-10, np.pi)))
    return factor[:, None] * (points - inner[:, None] * base[None, :])


def exp_map(base: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """Sphere exp-map at `base`. base [d], tangents [n, d] → points on sphere [n, d]."""
    norms = np.linalg.norm(tangents, axis=1)                            # [n]
    safe = np.clip(norms, 1e-10, None)
    direction = tangents / safe[:, None]
    return np.cos(norms)[:, None] * base[None, :] + np.sin(norms)[:, None] * direction


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--manifold-h", type=pathlib.Path, required=True,
                        help="M_h artifact: source of bin_edges and Q_max grid")
    parser.add_argument("--n-eps", type=int, default=100,
                        help="number of episodes (first N), must match M_h")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--smoothing", type=float, default=None,
                        help="UnivariateSpline `s`; default = n_used_bins")
    args = parser.parse_args()

    # ---- load ----
    logs = json.loads(args.dataset.read_text())[: args.n_eps]
    n_eps = len(logs)
    T = len(logs[0]["actions"])
    n_arms = len(logs[0]["payoff_matrix"][0])

    choice_dists = np.array([l["choice_dists"] for l in logs], dtype=np.float32)
    choice_dists = choice_dists[..., :n_arms]
    choice_dists = choice_dists / choice_dists.sum(axis=-1, keepdims=True)        # drop 'other'

    q_data = torch.load(args.q, weights_only=False)
    Q = q_data["Q"][:n_eps].cpu().numpy()                                          # [n_eps, T, n_arms]

    mh_data = torch.load(args.manifold_h, weights_only=False)
    bin_edges = mh_data["bin_edges"].cpu().numpy()
    q_grid = mh_data["manifold_q_grid"].cpu().numpy()
    n_grid = len(q_grid)

    # ---- skip trial 0; sort each trial's distribution by Q ordering (descending) ----
    choice_dists = choice_dists[:, 1:]
    Q_pred = Q[:, 1:]
    T_pred = T - 1

    rank_indices = np.argsort(-Q_pred, axis=-1)                                    # [n_eps, T_pred, n_arms]
    P_sorted = np.take_along_axis(choice_dists, rank_indices, axis=-1)             # [n_eps, T_pred, n_arms]
    Q_max = Q_pred.max(axis=-1)                                                    # [n_eps, T_pred]

    P_flat = P_sorted.reshape(-1, n_arms)                                          # [n_samples, n_arms]
    Q_flat = Q_max.reshape(-1)

    print(f"{n_eps} eps × {T_pred} trials = {len(Q_flat)} samples; n_arms={n_arms}")
    print("sorted-distribution marginals:")
    print(f"  rank 0 (best Q):  mean P = {P_flat[:, 0].mean():.3f}")
    print(f"  rank 1:           mean P = {P_flat[:, 1].mean():.3f}")
    print(f"  rank 2:           mean P = {P_flat[:, 2].mean():.3f}")
    print(f"  rank 3 (worst Q): mean P = {P_flat[:, 3].mean():.3f}")

    # ---- bin by Q_max using M_h's bin_edges ----
    n_bins = len(bin_edges) - 1
    # searchsorted to assign each sample to a bin (matches M_h's bucketize semantics)
    bin_idx = np.searchsorted(bin_edges[1:-1], Q_flat, side="right")
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    centroids: list[np.ndarray] = []
    bin_counts: list[int] = []
    bin_centers_used: list[float] = []
    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        if c > 0:
            centroids.append(P_flat[mask].mean(axis=0))
            bin_counts.append(c)
            bin_centers_used.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
    n_used = len(centroids)
    centroids_dist = np.stack(centroids)                                           # [n_used, n_arms]
    weights = np.sqrt(np.array(bin_counts, dtype=np.float64))
    x_centroids = np.array(bin_centers_used, dtype=np.float64)
    print(
        f"\nbinning: {n_bins} bins from M_h → {n_used} non-empty; "
        f"counts min/median/max: {min(bin_counts)}/{int(np.median(bin_counts))}/{max(bin_counts)}"
    )

    # ---- Hellinger embed → tangent at b_* → splines ----
    sqrt_centroids = np.sqrt(centroids_dist)                                       # [n_used, n_arms]; on sphere
    b_star = sqrt_centroids.mean(axis=0)
    b_star = b_star / np.linalg.norm(b_star)
    tangents = log_map(b_star, sqrt_centroids)                                     # [n_used, n_arms]

    smoothing = float(args.smoothing) if args.smoothing is not None else float(n_used)
    splines = [
        UnivariateSpline(x_centroids, tangents[:, d], w=weights, s=smoothing, k=3)
        for d in range(n_arms)
    ]

    # ---- sample on the Q_max grid: tangent → sphere → distribution ----
    tangent_grid = np.stack([spl(q_grid) for spl in splines], axis=1)              # [n_grid, n_arms]
    sphere_grid = exp_map(b_star, tangent_grid)                                    # [n_grid, n_arms]
    dist_grid = sphere_grid ** 2
    dist_grid = dist_grid / dist_grid.sum(axis=1, keepdims=True)                   # numerical safety

    # ---- diagnostic: spline fit quality at centroid bin positions ----
    fit_tangents = np.stack([spl(x_centroids) for spl in splines], axis=1)
    fit_sphere = exp_map(b_star, fit_tangents)
    fit_dist = fit_sphere ** 2
    fit_dist = fit_dist / fit_dist.sum(axis=1, keepdims=True)
    h_resid = (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(fit_dist) - sqrt_centroids, axis=1)
    print(
        f"spline fit (s={smoothing:.0f}): per-centroid Hellinger residual: "
        f"mean {float(h_resid.mean()):.4f}, max {float(h_resid.max()):.4f}"
    )

    # ---- save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sphere_grid":         torch.tensor(sphere_grid, dtype=torch.float32),
        "dist_grid":           torch.tensor(dist_grid, dtype=torch.float32),
        "hellinger_centroids": torch.tensor(sqrt_centroids, dtype=torch.float32),
        "dist_centroids":      torch.tensor(centroids_dist, dtype=torch.float32),
        "base_point":          torch.tensor(b_star, dtype=torch.float32),
        "tangent_centroids":   torch.tensor(tangents, dtype=torch.float32),
        "bin_centers":         torch.tensor(x_centroids, dtype=torch.float32),
        "bin_counts":          torch.tensor(bin_counts, dtype=torch.long),
        "bin_edges":           torch.tensor(bin_edges, dtype=torch.float32),
        "manifold_q_grid":     torch.tensor(q_grid, dtype=torch.float32),
        "n_arms":              n_arms,
        "n_grid":              n_grid,
        "smoothing":           smoothing,
        "model_id":            mh_data["model_id"],
        "value_model":         mh_data["value_model"],
        "n_eps":               n_eps,
    }, args.out)
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
