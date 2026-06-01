"""Visualize the fitted activation manifold M_h, in **centroid-PCA** coordinates.

Why a second PCA on the centroids? The first PCA (on all activations) captures the
top-variance directions of the *raw* residual stream — most of which is unrelated to
Q_max (letter identity, history, etc.). To see the manifold's *shape*, we want the
directions of largest variance *across centroids* — which by construction align with
the intrinsic coordinate (Q_max).

2 × 2 figure:
  Top-left  — 3D centroid-PCA of centroids (size ∝ bin count) + γ_M spline,
              coloured by Q_max (analog of Fig 5c's middle row, no waypoints yet).
  Top-right — 2D PC1–PC2 view (cleanest read of the curve shape).
  Bottom-L  — bin coverage (trials per Q_max bin).
  Bottom-R  — cumulative variance across centroids: if PC1 ≫ 0.5, the manifold is
              essentially 1-D in activation space; tells us how clean Q_max-as-axis is.

    python -m figures.fig_manifold \
        --manifold artifacts/manifolds/llama8b_layer19_Qmax.pt \
        --out artifacts/figures/fig_manifold_Mh.png
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
    parser.add_argument("--manifold", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    data = torch.load(args.manifold, weights_only=False)
    centroids = data["centroids_pca"].numpy()                # [n_bins, pca_dim] in activation-PCA basis
    manifold = data["manifold_grid"].numpy()                  # [n_grid, pca_dim]
    q_centers = data["bin_centers"].numpy()
    q_grid = data["manifold_q_grid"].numpy()
    counts = data["bin_counts"].numpy()
    layer = int(data["layer"])
    n_used = len(q_centers)

    # ---- second PCA on the centroids themselves (the "manifold axis" view) ----
    cent_mean = centroids.mean(axis=0, keepdims=True)
    cent_centered = centroids - cent_mean
    _, S_c, Vh_c = np.linalg.svd(cent_centered, full_matrices=False)
    cent_var = (S_c ** 2) / (S_c ** 2).sum()
    n_cent_pcs = min(3, Vh_c.shape[0])
    cent_pcs = Vh_c[:n_cent_pcs]                              # [≤3, pca_dim]

    centroids_3d = cent_centered @ cent_pcs.T                 # [n_bins, ≤3]
    manifold_3d = (manifold - cent_mean) @ cent_pcs.T         # [n_grid, ≤3]

    # ---- figure ----
    fig = plt.figure(figsize=(12.5, 9.5))
    sizes = 20 + 80 * counts / counts.max()

    # 3D centroid-PCA
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax3.plot(manifold_3d[:, 0], manifold_3d[:, 1], manifold_3d[:, 2],
             color="black", lw=1.0, alpha=0.55)
    sc = ax3.scatter(
        centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2],
        c=q_centers, cmap="viridis", s=sizes,
        edgecolor="black", linewidth=0.4,
    )
    ax3.set_xlabel(f"Centroid-PC 1 ({100*cent_var[0]:.0f}%)")
    ax3.set_ylabel(f"PC 2 ({100*cent_var[1]:.0f}%)")
    ax3.set_zlabel(f"PC 3 ({100*cent_var[2]:.0f}%)")
    ax3.set_title(f"M_h (3D centroid-PCA) — layer {layer}", fontsize=10)
    fig.colorbar(sc, ax=ax3, label="Q_max", shrink=0.6, pad=0.1)

    # 2D PC1-PC2
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(manifold_3d[:, 0], manifold_3d[:, 1], color="black", lw=1.0, alpha=0.55,
             label=f"γ_M spline ({len(q_grid)}-pt grid)")
    sc2 = ax2.scatter(
        centroids_3d[:, 0], centroids_3d[:, 1], c=q_centers, cmap="viridis", s=sizes,
        edgecolor="black", linewidth=0.4, label=f"{n_used} centroids",
    )
    ax2.set_xlabel(f"Centroid-PC 1 ({100*cent_var[0]:.0f}%)")
    ax2.set_ylabel(f"Centroid-PC 2 ({100*cent_var[1]:.0f}%)")
    ax2.set_title("M_h (centroid-PC1 vs PC2)", fontsize=10)
    ax2.legend(loc="best", fontsize=8)
    fig.colorbar(sc2, ax=ax2, label="Q_max")

    # Bin coverage
    axc = fig.add_subplot(2, 2, 3)
    # Use bar widths from spacing between consecutive bin centers (works for quantile bins too)
    if len(q_centers) > 1:
        spacings = np.diff(q_centers)
        bar_w = np.concatenate([spacings, spacings[-1:]]) * 0.9
    else:
        bar_w = np.array([1.0])
    axc.bar(q_centers, counts, width=bar_w, color="#6680a0",
            edgecolor="black", lw=0.3, align="center")
    axc.set_xlabel("Q_max (bin center)")
    axc.set_ylabel("Trials per bin")
    axc.set_title(
        f"Bin coverage — {n_used} bins, total {int(counts.sum())} trials  "
        f"(min/median/max count: {int(counts.min())}/{int(np.median(counts))}/{int(counts.max())})",
        fontsize=9,
    )

    # Centroid-PCA variance
    axv = fig.add_subplot(2, 2, 4)
    cum = cent_var.cumsum()
    n_show = min(15, len(cent_var))
    axv.plot(range(1, n_show + 1), cum[:n_show], marker="o", color="black", markersize=4)
    axv.fill_between(range(1, n_show + 1), 0, cum[:n_show], alpha=0.15, color="black")
    axv.axhline(0.5, color="gray", ls=":", lw=0.6)
    axv.axhline(0.9, color="gray", ls=":", lw=0.6)
    axv.set_xlabel("Centroid-PC index")
    axv.set_ylabel("Cumulative variance across centroids")
    one_d_score = float(cent_var[0])
    shape_call = "~1-D" if one_d_score > 0.5 else ("~2-D" if cent_var[:2].sum() > 0.7 else "higher-D")
    axv.set_title(
        f"Manifold shape: PC1 = {100*one_d_score:.0f}% of centroid variance → "
        f"{shape_call}", fontsize=10,
    )
    axv.set_ylim(0, 1.0)

    fig.suptitle(
        f"{data['model_id']}  layer {layer}  —  activation manifold over "
        f"{data['value_model']} Q_max",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved → {args.out}")
    print(f"  centroid-PCA: PC1={100*cent_var[0]:.1f}%, "
          f"PC1+2={100*cent_var[:2].sum():.1f}%, PC1+2+3={100*cent_var[:3].sum():.1f}%")


if __name__ == "__main__":
    main()
