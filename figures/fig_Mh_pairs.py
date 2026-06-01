"""Visualize the activation manifold M_h for each arm pair (PCA-3D).

Each panel shows, in the top-3 PCA dims of the layer's activation space:
  - the M_h spline (colored by r = Q[j]-Q[i], the intrinsic coordinate),
  - the binned activation centroids it interpolates (black dots),
  - the straight chord between endpoints (dashed) — the linear-steering path,
so the curvature that makes manifold ≠ linear is directly visible.

    python -m figures.fig_Mh_pairs \
        --pairs artifacts/manifolds/llama8b_fixed_pairs_L19.pt \
        --out artifacts/figures/fig_Mh_pairs_L19.png
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    mf = torch.load(args.pairs, weights_only=False)
    pairs = mf["pairs"]
    var = mf["pca_var_explained"].numpy()
    L = mf["layer"]
    pcs_var = var[:3].sum()

    keys = list(pairs)
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig = plt.figure(figsize=(5 * ncol, 4.4 * nrow))

    for idx, pk in enumerate(keys):
        pd = pairs[pk]
        grid = pd["manifold_grid"].numpy()[:, :3]
        cen = pd["centroids_pca"].numpy()[:, :3]
        r_grid = pd["r_grid"].numpy()
        li, lj = pd["letters"]

        # arc/chord for the title
        arc = np.linalg.norm(np.diff(grid, axis=0), axis=1).sum()
        chord = np.linalg.norm(grid[-1] - grid[0]) + 1e-12

        ax = fig.add_subplot(nrow, ncol, idx + 1, projection="3d")
        sc = ax.scatter(*grid.T, c=r_grid, cmap="viridis", s=12, zorder=2)
        ax.plot(*grid.T, color="gray", lw=1, alpha=0.5, zorder=1)
        ax.scatter(*cen.T, color="k", s=18, alpha=0.6, zorder=3, label="centroids")
        ax.plot(*np.stack([grid[0], grid[-1]]).T, color="#e8590c", lw=2, ls="--",
                zorder=4, label="linear chord")
        # endpoint markers
        ax.scatter(*grid[0], color="#1f6feb", s=80, marker="D", zorder=5)
        ax.scatter(*grid[-1], color="#c92a2a", s=80, marker="D", zorder=5)
        ax.text2D(0.02, 0.95, f"{li}→{lj}   arc/chord={arc/chord:.2f}",
                  transform=ax.transAxes, fontsize=10, weight="bold")
        ax.text2D(0.02, 0.02, f"◆ {li} (r min)   ◆ {lj} (r max)",
                  transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
        ax.set_zlabel("PC3", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Activation manifold $M_h$ per arm pair (layer {L}, PCA-3D; "
                 f"PC1-3 capture {pcs_var:.0%} of variance)", fontsize=13)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    fig.colorbar(sc, cax=cax, label="r = Q[j] − Q[i]")
    fig.subplots_adjust(right=0.91, hspace=0.1, wspace=0.05, top=0.93)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
