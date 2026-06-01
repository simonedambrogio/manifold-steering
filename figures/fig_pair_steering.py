"""Headline figure for V2 between-arms steering: manifold vs linear.

Four panels, echoing the paper's layout:
  A  Activation space (PCA-3D): M_h spline, straight chord, waypoints colored by r=Q[j]-Q[i].
  B1 Behavior space (ternary P(i)/P(j)/P(rest)): M_y curve + manifold + linear trajectories.
  B2 Behavior space (Hellinger-2D): the same, in the paper's sqrt-prob embedding.
  C  P(j) vs path fraction (manifold / linear / M_y reference) + distance-to-M_y energy inset.

    python -m figures.fig_pair_steering \
        --pairs artifacts/manifolds/llama8b_fixed_pairs_L19.pt \
        --steering artifacts/steering/llama8b_fixed_pair_steering.pt \
        --out artifacts/figures/fig_pair_steering.png            # --pair 0_3 to pick
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

C_MAN, C_LIN, C_MY = "#1f6feb", "#e8590c", "#444444"


def ternary_xy(p_i, p_j, p_rest):
    """Barycentric (i, j, rest) → 2D. i=bottom-left, j=bottom-right, rest=top."""
    s = p_i + p_j + p_rest + 1e-12
    p_i, p_j, p_rest = p_i / s, p_j / s, p_rest / s
    x = 0.5 * p_rest + p_j
    y = (np.sqrt(3) / 2) * p_rest
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    ap.add_argument("--steering", type=pathlib.Path, required=True)
    ap.add_argument("--pair", default=None, help="e.g. 0_3; default = largest manifold-vs-linear gap")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    mf = torch.load(args.pairs, weights_only=False)
    st = torch.load(args.steering, weights_only=False)
    res = st["results"]
    n_arms = st["n_arms"]
    lin_mode = "linear_full" if "linear_full" in st["modes"] else \
        next(m for m in st["modes"] if m.startswith("linear"))

    # choose showcase pair
    if args.pair and args.pair in res:
        pk = args.pair
    else:
        pk = max(res, key=lambda k: res[k][f"ebc_{lin_mode}"].numpy().mean()
                 - res[k]["ebc_manifold"].numpy().mean())
    R = res[pk]
    P = mf["pairs"][pk]
    i, j = R["i"], R["j"]
    li, lj = R["letters"]
    rest = [a for a in range(n_arms) if a not in (i, j)]
    K = st["K"]
    frac = np.linspace(0, 1, K)
    print(f"showcase pair {pk}  ({li}→{lj})")

    # mean trajectories over bases, renormalized over the 4 arms
    def traj(mode):
        d = res[pk][f"dist_{mode}"].numpy().mean(axis=0)        # [K, n_arms+1]
        arms = d[:, :n_arms] / d[:, :n_arms].sum(axis=1, keepdims=True)
        return arms, d[:, n_arms]                               # arm dist, 'other' mass
    man_arms, man_other = traj("manifold")
    lin_arms, lin_other = traj(lin_mode)
    dist_grid = P["dist_grid"].numpy()                          # M_y [G, n_arms]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # ---- A: activation space PCA-3D ----
    axA = fig.add_subplot(gs[0, 0], projection="3d")
    mg = P["manifold_grid"].numpy()[:, :3]
    cen = P["centroids_pca"].numpy()[:, :3]
    way = R["manifold_path_pca"].numpy()[:, :3]
    chord = R["linear_path_pca"].numpy()[:, :3]
    r_way = R["r_way"].numpy()
    axA.plot(*mg.T, color=C_MY, lw=2, label="$M_h$ spline", zorder=1)
    axA.plot(*chord.T, color=C_LIN, lw=2, ls="--", label="linear chord", zorder=2)
    sc = axA.scatter(*way.T, c=r_way, cmap="viridis", s=28, zorder=3)
    axA.scatter(*cen[[0, -1]].T, color="k", s=60, marker="D", zorder=4)
    axA.set_title(f"A  Activation space (PCA-3D)\n{li}→{lj}", fontsize=11, loc="left")
    axA.set_xlabel("PC1"); axA.set_ylabel("PC2"); axA.set_zlabel("PC3")
    axA.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axA, shrink=0.5, pad=0.1, label="r = Q[j]−Q[i]")

    # ---- C: P(j) vs path fraction + energy inset ----
    axC = fig.add_subplot(gs[0, 1])
    axC.plot(frac, man_arms[:, j], color=C_MAN, lw=2.5, label="manifold")
    axC.plot(frac, lin_arms[:, j], color=C_LIN, lw=2.5, ls="--", label="linear")
    # M_y reference: P(j) along the natural curve, parameterized to match endpoints
    axC.plot(np.linspace(0, 1, len(dist_grid)), dist_grid[:, j], color=C_MY, lw=1.5,
             alpha=0.6, label="$M_y$ (natural)")
    axC.plot(frac, man_other, color=C_MAN, lw=1, alpha=0.5)
    axC.plot(frac, lin_other, color=C_LIN, lw=1, ls=":", alpha=0.7, label="linear P(other)")
    axC.set_xlabel("path fraction (i → j)"); axC.set_ylabel(f"P(arm {lj})")
    axC.set_title("C  Choice mass along the path", fontsize=11, loc="left")
    axC.legend(fontsize=8); axC.set_ylim(-0.02, 1.0)
    # inset: cumulative distance-to-M_y
    axCi = axC.inset_axes([0.58, 0.16, 0.38, 0.34])
    man_e = res[pk]["ebc_manifold"].numpy()
    lin_e = res[pk][f"ebc_{lin_mode}"].numpy()
    axCi.bar([0, 1], [man_e.mean(), lin_e.mean()],
             yerr=[man_e.std()/np.sqrt(len(man_e)), lin_e.std()/np.sqrt(len(lin_e))],
             color=[C_MAN, C_LIN], width=0.6)
    axCi.set_xticks([0, 1]); axCi.set_xticklabels(["man", "lin"], fontsize=7)
    axCi.set_title("$E_{BC}$ (↓natural)", fontsize=7); axCi.tick_params(labelsize=6)

    # ---- B1: ternary ----
    axB1 = fig.add_subplot(gs[1, 0])
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    axB1.plot(tri[:, 0], tri[:, 1], color="lightgray", lw=1)
    for (px, py, txt) in [(-0.04, -0.03, f"{li}"), (1.02, -0.03, f"{lj}"),
                          (0.5, np.sqrt(3)/2 + 0.03, "rest")]:
        axB1.text(px, py, txt, fontsize=10, ha="center")
    def tern(arms):
        return ternary_xy(arms[:, i], arms[:, j], arms[:, rest].sum(axis=1))
    mx, my = ternary_xy(dist_grid[:, i], dist_grid[:, j], dist_grid[:, rest].sum(axis=1))
    axB1.plot(mx, my, color=C_MY, lw=2, label="$M_y$")
    axB1.plot(*tern(man_arms), color=C_MAN, lw=2.5, marker="o", ms=3, label="manifold")
    axB1.plot(*tern(lin_arms), color=C_LIN, lw=2.5, ls="--", marker="s", ms=3, label="linear")
    axB1.set_title("B1  Behavior space (ternary)", fontsize=11, loc="left")
    axB1.set_aspect("equal"); axB1.axis("off"); axB1.legend(fontsize=8, loc="upper right")

    # ---- B2: Hellinger-2D (PCA of sqrt-prob, fit on M_y + trajectories) ----
    axB2 = fig.add_subplot(gs[1, 1])
    anchor = np.sqrt(dist_grid)
    amean = anchor.mean(0)
    _, _, Vt = np.linalg.svd(anchor - amean, full_matrices=False)
    proj = lambda P4: (np.sqrt(P4) - amean) @ Vt[:2].T
    my2 = proj(dist_grid)
    axB2.plot(my2[:, 0], my2[:, 1], color=C_MY, lw=2, label="$M_y$")
    m2, l2 = proj(man_arms), proj(lin_arms)
    axB2.plot(m2[:, 0], m2[:, 1], color=C_MAN, lw=2.5, marker="o", ms=3, label="manifold")
    axB2.plot(l2[:, 0], l2[:, 1], color=C_LIN, lw=2.5, ls="--", marker="s", ms=3, label="linear")
    axB2.set_title("B2  Behavior space (Hellinger-2D)", fontsize=11, loc="left")
    axB2.set_xlabel("sqrt-prob PC1"); axB2.set_ylabel("sqrt-prob PC2")
    axB2.legend(fontsize=8)

    ratio = lin_e.mean() / man_e.mean()
    fig.suptitle(f"Manifold vs linear steering, {li}→{lj}  "
                 f"(E_BC {ratio:.1f}× more natural under manifold)", fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
