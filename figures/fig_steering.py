"""Steering figure (paper Fig 5c-style).

Three panels:

  Top — 3D centroid-PCA of M_h with both steering paths overlaid (manifold = blue solid,
        linear = red dashed) and K waypoints marked. Reveals that linear "cuts through"
        the manifold off-curve, while manifold steering stays on the curve.

  Bottom-left  — Behavior under MANIFOLD steering: P(rank 0..3) vs waypoint index.
                 Expect smooth, monotone change (Q-best arm gains/loses mass smoothly).
  Bottom-right — Behavior under LINEAR steering: same axes. May show non-monotone /
                 "teleporting" structure where linear cuts through off-manifold regions.

    python -m figures.fig_steering \
        --steering   artifacts/steering/llama8b_layer19_Qmax_steering.pt \
        --manifold-h artifacts/manifolds/llama8b_layer19_Qmax.pt \
        --out        artifacts/figures/fig_steering.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steering",   type=pathlib.Path, required=True)
    parser.add_argument("--manifold-h", type=pathlib.Path, required=True)
    parser.add_argument("--manifold-y", type=pathlib.Path, default=None,
                        help="overlay M_y's predicted P(rank) at each waypoint")
    parser.add_argument("--out",        type=pathlib.Path, required=True)
    args = parser.parse_args()

    st = torch.load(args.steering, weights_only=False)
    mh = torch.load(args.manifold_h, weights_only=False)
    my = torch.load(args.manifold_y, weights_only=False) if args.manifold_y else None

    centroids = mh["centroids_pca"].numpy()           # [n_bins, 64]
    manifold = mh["manifold_grid"].numpy()             # [n_grid, 64]
    q_centers = mh["bin_centers"].numpy()
    counts = mh["bin_counts"].numpy()

    manifold_path = st["manifold_path_pca"].numpy()    # [K, 64]
    linear_path = st["linear_path_pca"].numpy()
    q_waypoints = st["waypoint_q_values"].numpy()
    manifold_avg = st["manifold_dist_avg"].numpy()     # [K, n_arms]
    linear_avg = st["linear_dist_avg"].numpy()
    manifold_per = st["manifold_dist_per_base"].numpy()  # [P, K, n_arms]
    linear_per = st["linear_dist_per_base"].numpy()
    K = int(st["K"])
    P = manifold_per.shape[0]
    n_arms = manifold_avg.shape[1]

    # M_y prediction at each waypoint: look up nearest grid point on the behavior manifold
    my_pred = None  # [K, n_arms]
    if my is not None:
        my_dist_grid = my["dist_grid"].numpy()              # [n_grid_y, n_arms], rows = P(rank 0..3)
        my_q_grid = my["manifold_q_grid"].numpy()           # [n_grid_y]
        my_pred = np.stack([
            my_dist_grid[int(np.argmin(np.abs(my_q_grid - q)))] for q in q_waypoints
        ])                                                  # [K, n_arms]

    # Centroid-PCA basis for visualization (consistent with fig_manifold.py)
    cent_mean = centroids.mean(axis=0, keepdims=True)
    cent_centered = centroids - cent_mean
    _, _, Vh_c = np.linalg.svd(cent_centered, full_matrices=False)
    cent_pcs = Vh_c[:3]
    centroids_3d = cent_centered @ cent_pcs.T                       # [n_bins, 3]
    manifold_3d = (manifold - cent_mean) @ cent_pcs.T               # [n_grid, 3]
    manifold_path_3d = (manifold_path - cent_mean) @ cent_pcs.T     # [K, 3]
    linear_path_3d = (linear_path - cent_mean) @ cent_pcs.T         # [K, 3]

    fig = plt.figure(figsize=(13, 9.5))
    sizes = 20 + 70 * counts / counts.max()

    # ---- Top: 3D PCA with both paths ----
    ax3 = fig.add_subplot(2, 2, (1, 2), projection="3d")
    ax3.plot(manifold_3d[:, 0], manifold_3d[:, 1], manifold_3d[:, 2],
             color="black", lw=0.7, alpha=0.4)
    sc = ax3.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2],
                     c=q_centers, cmap="viridis", s=sizes,
                     edgecolor="black", linewidth=0.3, alpha=0.8)
    # Manifold steering path
    ax3.plot(manifold_path_3d[:, 0], manifold_path_3d[:, 1], manifold_path_3d[:, 2],
             color="#1f77b4", lw=2.0, alpha=0.9, label="manifold steering path")
    ax3.scatter(manifold_path_3d[:, 0], manifold_path_3d[:, 1], manifold_path_3d[:, 2],
                color="#1f77b4", s=50, edgecolor="black", linewidth=0.5)
    # Linear steering path
    ax3.plot(linear_path_3d[:, 0], linear_path_3d[:, 1], linear_path_3d[:, 2],
             color="#d62728", lw=2.0, ls="--", alpha=0.9, label="linear steering path")
    ax3.scatter(linear_path_3d[:, 0], linear_path_3d[:, 1], linear_path_3d[:, 2],
                color="#d62728", s=50, edgecolor="black", linewidth=0.5)
    ax3.set_xlabel("Centroid-PC 1")
    ax3.set_ylabel("Centroid-PC 2")
    ax3.set_zlabel("Centroid-PC 3")
    ax3.set_title(f"M_h with steering paths — layer {st['layer']}  ({K} waypoints, "
                  f"Q_max {st['q_low']:+.1f} → {st['q_high']:+.1f})", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax3, label="Q_max (centroids)", shrink=0.5, pad=0.1)

    # ---- Bottom-left: manifold-steering behavior ----
    rank_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_arms))[::-1]
    rank_labels = ["P(rank 0 = best Q arm)", "P(rank 1)", "P(rank 2)", "P(rank 3 = worst Q)"]

    def _plot_traj(ax, dist_avg, dist_per, title, ylabel=True):
        x = np.arange(K)
        for r in range(n_arms):
            # per-prompt as faint lines, average as bold line
            for p in range(P):
                ax.plot(x, dist_per[p, :, r], color=rank_colors[r],
                        alpha=0.15, lw=0.6)
            ax.plot(x, dist_avg[:, r], color=rank_colors[r], lw=2.2,
                    marker="o", markersize=4, label=rank_labels[r])
            # M_y prediction: what the cognitive model says P(rank r) should be
            if my_pred is not None:
                ax.plot(x, my_pred[:, r], color=rank_colors[r], lw=1.6,
                        ls="--", marker="s", markersize=3, alpha=0.85)
        ax.set_xlabel("Waypoint index (low Q_max → high Q_max)")
        if ylabel:
            ax.set_ylabel("P(arm | steered activation), Q-ranked")
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, K, max(1, K // 8)))
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)

    ax_m = fig.add_subplot(2, 2, 3)
    title_suffix = "  (solid = steered LLM, dashed = M_y prediction)" if my_pred is not None else ""
    _plot_traj(ax_m, manifold_avg, manifold_per,
               f"Manifold steering — behavior (avg of {P} prompts){title_suffix}")
    ax_m.legend(loc="center right", fontsize=7)

    ax_l = fig.add_subplot(2, 2, 4, sharey=ax_m)
    _plot_traj(ax_l, linear_avg, linear_per,
               f"Linear steering — behavior{title_suffix}", ylabel=False)

    fig.suptitle(
        f"Steering — {st['model_id']}  layer {st['layer']}  →  {st['value_model']} Q_max",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved → {args.out}")
    print(f"  steered manifold P(rank 0..3): {np.round(manifold_avg[0], 3)} → "
          f"{np.round(manifold_avg[-1], 3)}")
    print(f"  steered linear   P(rank 0..3): {np.round(linear_avg[0], 3)} → "
          f"{np.round(linear_avg[-1], 3)}")
    if my_pred is not None:
        print(f"  M_y prediction   P(rank 0..3): {np.round(my_pred[0], 3)} → "
              f"{np.round(my_pred[-1], 3)}")
        # How much of M_y's predicted swing did each strategy deliver?
        for name, swing in [("manifold", manifold_avg[-1] - manifold_avg[0]),
                            ("linear",   linear_avg[-1] - linear_avg[0])]:
            target = my_pred[-1] - my_pred[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(np.abs(target) > 1e-4, swing / target, np.nan)
            print(f"  {name:<8} delivered/predicted Δ per rank: {np.round(frac, 2)}")


if __name__ == "__main__":
    main()
