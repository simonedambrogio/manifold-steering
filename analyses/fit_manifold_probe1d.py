"""Fit a 1-D activation manifold M_h along the *probe direction* for a target scalar.

Drop-in variant of fit_manifold.py. Instead of using the top-K PCA directions
(unsupervised, maximises general activation variance), this script uses the
**ridge probe weight vector** for the target scalar as the 1-D subspace
(supervised, maximises target-variable predictability).

This is option 1 of the subspace-misalignment fix: collapse the manifold to the
single direction the probe found, where 100% of the Q_max signal lives by
construction. If steering along *this* direction reproduces M_y's predicted swing,
we've confirmed the misalignment was the whole reason the original steering failed.

Output (same schema as fit_manifold.py so it's a drop-in for steer.py / fig):
  centroids_pca       [n_used_bins, 1]   — scalar projection per bin
  manifold_grid       [n_grid, 1]
  pca_basis           [1, hidden_dim]    — unit-normed probe direction
  pca_mean            [hidden_dim]
  (... plus bin_centers, bin_counts, bin_edges, manifold_q_grid, layer, ...)

    python -m analyses.fit_manifold_probe1d \
        --activations artifacts/activations/llama8b_residual.pt \
        --q           artifacts/value/bestrl_Q.pt \
        --probe       artifacts/locate/llama8b_bestrl_scalars.pt \
        --target Q_max --layer 19 --n-bins 30 \
        --out         artifacts/manifolds/llama8b_layer19_Qmax_probe1d.pt
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
    parser.add_argument("--q",           type=pathlib.Path, required=True)
    parser.add_argument("--probe",       type=pathlib.Path, required=True,
                        help="locate.py .pt with `betas`, `train_stds`, `target_names`")
    parser.add_argument("--target",      default="Q_max")
    parser.add_argument("--layer",       type=int, required=True)
    parser.add_argument("--n-bins",      type=int, default=30)
    parser.add_argument("--binning",     choices=["equal", "quantile"], default="quantile")
    parser.add_argument("--n-grid",      type=int, default=200)
    parser.add_argument("--smoothing",   type=float, default=None,
                        help="UnivariateSpline `s`; default = n_used_bins")
    parser.add_argument("--out",         type=pathlib.Path, required=True)
    args = parser.parse_args()

    # ---- load activations + Q ----
    act_data = torch.load(args.activations, weights_only=False)
    activations = act_data["activations"]                  # [n_eps, T, n_layers, hidden_dim]
    ep_ids = act_data["episode_ids"]
    n_eps, T, n_layers, hidden_dim = activations.shape

    q_data = torch.load(args.q, weights_only=False)
    Q = q_data["Q"][ep_ids]                                # [n_eps, T, n_arms]

    # Skip trial 0 (uninformative: Q[0] uniform init)
    A = activations[:, 1:, args.layer, :].float()          # [n_eps, T-1, hidden_dim]
    Q_max = Q[:, 1:].max(dim=-1).values                    # [n_eps, T-1]
    A_flat = A.reshape(-1, hidden_dim)
    Q_flat = Q_max.reshape(-1)
    n_samples = A_flat.shape[0]
    print(f"layer {args.layer}: {n_samples} samples, "
          f"Q_max range [{float(Q_flat.min()):+.1f}, {float(Q_flat.max()):+.1f}]")

    # ---- load probe; convert from standardized-space to original-activation-space ----
    probe = torch.load(args.probe, weights_only=False)
    target_names = probe["target_names"]
    if args.target not in target_names:
        raise SystemExit(f"target {args.target!r} not in {target_names}")
    t_idx = target_names.index(args.target)
    beta_std = probe["betas"][args.layer, :, t_idx]        # [hidden_dim], standardized-space weights
    train_sd = probe["train_stds"][args.layer]             # [hidden_dim]
    w_orig = beta_std / train_sd                           # [hidden_dim], raw-activation-space direction
    direction = w_orig / w_orig.norm()                     # [hidden_dim], unit-norm
    print(f"probe direction for {args.target} at layer {args.layer}: "
          f"||w_orig||={float(w_orig.norm()):.2f} → normalized")

    A_mean = A_flat.mean(dim=0)                            # [hidden_dim]
    A_centered = A_flat - A_mean

    # 1-D projection along the probe direction (unit basis row → [1, hidden_dim])
    pca_basis = direction.unsqueeze(0)                     # [1, hidden_dim]
    A_proj = A_centered @ pca_basis.T                       # [n_samples, 1]
    proj_var = float(A_proj.var())
    total_var = float(A_centered.pow(2).sum() / n_samples)
    print(f"variance along probe direction: {proj_var:.3f} "
          f"(= {proj_var/total_var*100:.3f}% of total per-dim variance)")

    # ---- bin by Q_max (identical to fit_manifold.py) ----
    q_min, q_max = float(Q_flat.min()), float(Q_flat.max())
    if args.binning == "quantile":
        qs = np.linspace(0.0, 1.0, args.n_bins + 1)
        edges_np = np.unique(np.quantile(Q_flat.cpu().numpy(), qs))
        bin_edges = torch.tensor(edges_np, dtype=torch.float32)
    else:
        bin_edges = torch.linspace(q_min, q_max, args.n_bins + 1)
    n_actual_bins = len(bin_edges) - 1
    bin_idx = torch.bucketize(Q_flat, bin_edges[1:-1], right=False).clamp(max=n_actual_bins - 1)

    centroids: list[torch.Tensor] = []
    bin_centers_used: list[float] = []
    bin_counts: list[int] = []
    for b in range(n_actual_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        if c > 0:
            centroids.append(A_proj[mask].mean(dim=0))     # [1]
            bin_centers_used.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
            bin_counts.append(c)
    n_used = len(centroids)
    centroids_arr = torch.stack(centroids)                  # [n_used, 1]
    x_centroids = np.array(bin_centers_used, dtype=np.float64)
    weights = np.sqrt(np.array(bin_counts, dtype=np.float64))

    print(f"bins: {n_used} non-empty; counts min/max/median = "
          f"{min(bin_counts)}/{max(bin_counts)}/{int(np.median(bin_counts))}")

    # ---- fit smoothing spline (just one — single dimension) ----
    smoothing = float(args.smoothing) if args.smoothing is not None else float(n_used)
    y = centroids_arr[:, 0].cpu().numpy().astype(np.float64)
    spl = UnivariateSpline(x_centroids, y, w=weights, s=smoothing, k=3)

    q_grid = np.linspace(q_min, q_max, args.n_grid)
    manifold_grid = spl(q_grid)[:, None]                    # [n_grid, 1]

    residuals = spl(x_centroids) - y
    rmse = float(np.sqrt((residuals ** 2 * weights).sum() / weights.sum()))
    centroid_range = float(y.max() - y.min())
    print(f"spline RMSE / centroid-range = {rmse:.3f} / {centroid_range:.3f} "
          f"= {rmse/centroid_range*100:.1f}%")
    print(f"manifold coord swing across Q_max range: "
          f"{float(manifold_grid[0,0]):+.3f} → {float(manifold_grid[-1,0]):+.3f}")

    # ---- save (same schema as fit_manifold.py) ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "centroids_pca":     centroids_arr.cpu(),                                   # [n_used, 1]
        "bin_centers":       torch.tensor(x_centroids, dtype=torch.float32),
        "bin_counts":        torch.tensor(bin_counts, dtype=torch.long),
        "bin_edges":         bin_edges,
        "manifold_grid":     torch.tensor(manifold_grid, dtype=torch.float32),      # [n_grid, 1]
        "manifold_q_grid":   torch.tensor(q_grid, dtype=torch.float32),
        "pca_basis":         pca_basis.cpu(),                                       # [1, hidden_dim] — probe direction
        "pca_mean":          A_mean.cpu(),                                          # [hidden_dim]
        "pca_var_explained": torch.tensor([proj_var / total_var], dtype=torch.float32),
        "layer":             args.layer,
        "n_bins":            args.n_bins,
        "n_grid":            args.n_grid,
        "smoothing":         smoothing,
        "q_min":             q_min,
        "q_max":             q_max,
        "model_id":          act_data["model_id"],
        "value_model":       q_data["model_name"],
        "subspace_kind":     f"probe_direction:{args.target}",
        "probe_target":      args.target,
    }, args.out)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
