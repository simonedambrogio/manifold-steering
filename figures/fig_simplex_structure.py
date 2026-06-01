"""Structural correspondence of the 4-arm simplex: activation space ↔ behavior space.

The bandit's concept structure is a simplex — 4 "arm-X-is-best" vertices joined by 6
value-transition edges. We show that this same structure appears in both:
  Left   activation space (M_h), top-3 PCA dims.
  Right  behavior space, exact barycentric embedding into a regular tetrahedron (each
         4-arm distribution placed by its probabilities; no MDS, no distortion).

This is a QUALITATIVE correspondence panel. The quantitative isometry does NOT favor
manifold over linear here (within-edge Pearson r≈0.975 for both), because the value-
transition edges are curved-but-non-folding — unlike the paper's cyclic weekday loop,
where folding is what makes linear distance fail. Annotated on the figure.

    python -m figures.fig_simplex_structure \
        --simplex artifacts/manifolds/llama8b_fixed_simplex_L19.pt \
        --out artifacts/figures/fig_simplex_structure_L19.png
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ARM_COLORS = ["#1f6feb", "#e8590c", "#2f9e44", "#9c36b5"]   # B, C, D, F


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simplex", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    d = torch.load(args.simplex, weights_only=False)
    H = d["nodes_h"].numpy()                      # [n, pca_dim] activation
    Y = d["nodes_y"].numpy()                      # [n, n_arms] behavior dist
    letters = d["letters"]
    n_arms = d["n_arms"]
    edge_specs = d["edge_specs"]                   # (i, j, s, e, pk)
    # within-edge isometry (the honest, artifact-free number)
    from scipy.stats import pearsonr
    Dm, Dl, Dy = d["D_h_manifold"].numpy(), d["D_h_linear"].numpy(), d["D_y"].numpy()
    nt = d["node_type"]
    iu = np.triu_indices(len(nt), k=1)
    within = np.array([nt[a] == nt[b] and nt[a] != "vertex" for a, b in zip(*iu)])
    r_man = pearsonr(Dm[iu][within], Dy[iu][within])[0]
    r_lin = pearsonr(Dl[iu][within], Dy[iu][within])[0]

    # regular tetrahedron corners for the behavior simplex
    T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)
    Y3 = Y @ T                                     # [n, 3] barycentric placement
    H3 = H[:, :3]                                  # activation PCA-3D

    fig = plt.figure(figsize=(14, 6.5))
    axH = fig.add_subplot(1, 2, 1, projection="3d")
    axY = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, P3, title in [(axH, H3, "Activation space  $M_h$  (PCA-3D)"),
                          (axY, Y3, "Behavior space  (4-arm simplex)")]:
        # edges
        for (i, j, s, e, pk) in edge_specs:
            seg = P3[s:e + 1]
            ax.plot(*seg.T, color="gray", lw=1.6, alpha=0.7, zorder=1)
        # vertices
        for X in range(n_arms):
            ax.scatter(*P3[X], color=ARM_COLORS[X], s=160, marker="D",
                       edgecolor="k", linewidth=0.5, zorder=5)
            ax.text(*P3[X], f"  {letters[X]}", fontsize=13, weight="bold", zorder=6)
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)

    fig.suptitle("4-arm simplex: the same vertex+edge structure in activation and behavior space",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.04,
             f"Vertices = 'arm-X-is-best' states; edges = value-transition manifolds. "
             f"Qualitative correspondence only: within-edge isometry manifold r={r_man:.2f} "
             f"≈ linear r={r_lin:.2f} per within-edge pairs — the value edges are curved but "
             f"non-folding, so (unlike cyclic concepts) distance does not separate the two.",
             ha="center", fontsize=8.5, wrap=True)
    fig.subplots_adjust(top=0.9, bottom=0.12, wspace=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
