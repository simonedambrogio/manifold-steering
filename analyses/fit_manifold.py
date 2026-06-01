"""Fit the 1-D activation manifold M_h over Q_max at a chosen layer.

Recipe (mirrors mountain-car App. A):
  1. Bin trials by Q_max (intrinsic coordinate).
  2. Per-bin centroid: mean residual-stream activation over trials in the bin.
  3. PCA-reduce all activations to `--pca-dim` dimensions (default 64).
  4. Fit one univariate smoothing spline per PCA coordinate through the centroids,
     weighted by sqrt(bin_count) to handle sparse bins.
  5. Pre-sample the spline on a fine Q_max grid → `manifold_grid` for fast lookup.

Output (`artifacts/manifolds/<name>.pt`):
  centroids_pca       [n_used_bins, pca_dim]
  bin_centers         [n_used_bins]      — Q_max at each bin's centre
  bin_counts          [n_used_bins]
  bin_edges           [n_bins + 1]
  manifold_grid       [n_grid, pca_dim]  — γ_M sampled
  manifold_q_grid     [n_grid]
  pca_basis           [pca_dim, hidden_dim]  — V^T from SVD
  pca_mean            [hidden_dim]
  pca_var_explained   [pca_dim]
  layer, n_bins, n_grid, smoothing, q_min, q_max, model_id, value_model

    python -m analyses.fit_manifold \
        --activations artifacts/activations/llama8b_residual.pt \
        --q artifacts/value/bestrl_Q.pt \
        --layer 19 --n-bins 30 --pca-dim 64 \
        --out artifacts/manifolds/llama8b_layer19_Qmax.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--activations", type=pathlib.Path, required=True)
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--binning", choices=["equal", "quantile"], default="quantile",
                        help="equal-width vs equal-count (quantile) bin edges over Q_max")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--smoothing", type=float, default=None,
        help="UnivariateSpline `s`; default = n_used_bins (mild smoothing)",
    )
    args = parser.parse_args()

    # ---- load activations + Q ----
    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]                  # [n_eps, T, n_layers, hidden_dim]
    ep_ids = act_data["episode_ids"]
    n_eps, T, n_layers, hidden_dim = activations.shape

    q_data = torch.load(args.q, weights_only=False)
    Q = q_data["Q"][ep_ids]                                # [n_eps, T, n_arms]

    # Skip trial 0; pick the chosen layer
    A = activations[:, 1:, args.layer, :].float()          # [n_eps, T-1, hidden_dim]
    Q_max = Q[:, 1:].max(dim=-1).values                    # [n_eps, T-1]

    A_flat = A.reshape(-1, hidden_dim)                     # [n_samples, hidden_dim]
    Q_flat = Q_max.reshape(-1)                             # [n_samples]
    n_samples = A_flat.shape[0]
    print(
        f"layer {args.layer}: {n_samples} samples (= {n_eps} eps × {T-1} trials), "
        f"Q_max range [{float(Q_flat.min()):+.1f}, {float(Q_flat.max()):+.1f}]"
    )

    # ---- PCA on all activations (fit a basis for the activation space) ----
    A_mean = A_flat.mean(dim=0)
    A_centered = A_flat - A_mean
    _, S, Vh = torch.linalg.svd(A_centered, full_matrices=False)
    pca_basis = Vh[: args.pca_dim]                         # [pca_dim, hidden_dim]
    var_explained = (S ** 2) / (S ** 2).sum()
    print(
        f"PCA top {args.pca_dim}: cumulative var explained = "
        f"{float(var_explained[: args.pca_dim].sum()):.3f}; "
        f"PC1 = {float(var_explained[0]):.3f}"
    )

    # Project all samples into PCA space
    A_pca = A_centered @ pca_basis.T                        # [n_samples, pca_dim]

    # ---- bin by Q_max ----
    q_min, q_max = float(Q_flat.min()), float(Q_flat.max())
    if args.binning == "quantile":
        # Equal-count bins; dedupe in case Q_flat has ties at quantile boundaries.
        qs = np.linspace(0.0, 1.0, args.n_bins + 1)
        edges_np = np.unique(np.quantile(Q_flat.cpu().numpy(), qs))
        bin_edges = torch.tensor(edges_np, dtype=torch.float32)
    else:
        bin_edges = torch.linspace(q_min, q_max, args.n_bins + 1)
    n_actual_bins = len(bin_edges) - 1
    bin_idx = torch.bucketize(Q_flat, bin_edges[1:-1], right=False).clamp(max=n_actual_bins - 1)
    print(f"binning: {args.binning}; requested {args.n_bins} bins → {n_actual_bins} edges")

    centroids: list[torch.Tensor] = []
    bin_centers_used: list[float] = []
    bin_counts: list[int] = []
    for b in range(n_actual_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        if c > 0:
            centroids.append(A_pca[mask].mean(dim=0))
            bin_centers_used.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
            bin_counts.append(c)
    n_used = len(centroids)
    centroids_arr = torch.stack(centroids)                  # [n_used, pca_dim]
    x_centroids = np.array(bin_centers_used, dtype=np.float64)
    weights = np.sqrt(np.array(bin_counts, dtype=np.float64))

    print(
        f"bins: {args.n_bins} requested → {n_used} non-empty; "
        f"counts: min={min(bin_counts)}, max={max(bin_counts)}, "
        f"median={int(np.median(bin_counts))}"
    )

    # ---- fit smoothing spline per PCA coordinate ----
    smoothing = float(args.smoothing) if args.smoothing is not None else float(n_used)
    splines = []
    for d in range(args.pca_dim):
        y = centroids_arr[:, d].cpu().numpy().astype(np.float64)
        spl = UnivariateSpline(x_centroids, y, w=weights, s=smoothing, k=3)
        splines.append(spl)

    # ---- pre-sample the manifold on a fine Q_max grid ----
    q_grid = np.linspace(q_min, q_max, args.n_grid)
    manifold_grid = np.stack([spl(q_grid) for spl in splines], axis=1)   # [n_grid, pca_dim]

    # Diagnostic: weighted residuals at the centroid points
    fit_at_centers = np.stack([spl(x_centroids) for spl in splines], axis=1)  # [n_used, pca_dim]
    residuals = fit_at_centers - centroids_arr.cpu().numpy()
    rmse_per_dim = np.sqrt((residuals ** 2 * weights[:, None]).sum(axis=0) / weights.sum())
    print(
        f"spline fit (smoothing s={smoothing:.0f}): "
        f"per-PC weighted RMSE: median {float(np.median(rmse_per_dim)):.3f}, "
        f"max {float(rmse_per_dim.max()):.3f}"
    )

    # ---- save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "centroids_pca":     centroids_arr.cpu(),
        "bin_centers":       torch.tensor(x_centroids, dtype=torch.float32),
        "bin_counts":        torch.tensor(bin_counts, dtype=torch.long),
        "bin_edges":         bin_edges,
        "manifold_grid":     torch.tensor(manifold_grid, dtype=torch.float32),
        "manifold_q_grid":   torch.tensor(q_grid, dtype=torch.float32),
        "pca_basis":         pca_basis.cpu(),
        "pca_mean":          A_mean.cpu(),
        "pca_var_explained": var_explained[: args.pca_dim].cpu(),
        "layer":             args.layer,
        "n_bins":            args.n_bins,
        "n_grid":            args.n_grid,
        "smoothing":         smoothing,
        "q_min":             q_min,
        "q_max":             q_max,
        "model_id":          act_data["model_id"],
        "value_model":       q_data["model_name"],
    }, args.out)

    print(f"saved → {args.out}")
    print(
        f"  centroids_pca: {tuple(centroids_arr.shape)}; "
        f"manifold_grid: {tuple(manifold_grid.shape)}"
    )


if __name__ == "__main__":
    main()
