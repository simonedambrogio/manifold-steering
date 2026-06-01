"""Paper-style MDS-of-distances side-by-side embedding for the 4-arm simplex.

Reads the precomputed isometry artifact (`analyses.isometry_simplex`) and renders three
classical-MDS embeddings of the SAME node set (4 vertices + 6 edges) under three distance
metrics, Procrustes-aligned to the behavior embedding so orientations match:

  Behavior (D_y)            — Hellinger distance between choice distributions (the reference)
  Activation manifold (D_h) — geodesic distance along the simplex edges
  Activation linear         — straight-line Euclidean distance in PCA-64 (what linear sees)

If activation geometry recapitulates behavior geometry, its MDS recovers the same tetrahedron.

    .venv/bin/python -m figures.isometry_mds \
        --simplex artifacts/manifolds/llama8b_fixed_simplex_L19.pt --outdir bandit/figures
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

ARM_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
EDGE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def procrustes(X, Y):
    """Rotate/reflect X onto Y (orthogonal Procrustes, no scaling). X,Y [n,3] → aligned X."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    return Xc @ (U @ Vt) + Y.mean(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simplex", type=pathlib.Path,
                   default=pathlib.Path("artifacts/manifolds/llama8b_fixed_simplex_L19.pt"))
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    d = torch.load(args.simplex, weights_only=False)
    letters = d["letters"]; n_arms = d["n_arms"]
    edge_specs = d["edge_specs"]
    mds = {k: v.numpy() for k, v in d["mds"].items()}
    ref = mds["behavior"]
    panels = [
        ("Behavior  (D_y, Hellinger)", ref, None),
        (f"Activation manifold  (geodesic)   r={d['r_manifold']:.3f}",
         procrustes(mds["activation_manifold"], ref), d["r_manifold"]),
        (f"Activation linear  (Euclidean)   r={d['r_linear']:.3f}",
         procrustes(mds["activation_linear"], ref), d["r_linear"]),
    ]

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=[t for t, _, _ in panels], horizontal_spacing=0.02)
    for col, (title, P, _) in enumerate(panels, start=1):
        # vertices
        for x in range(n_arms):
            fig.add_trace(go.Scatter3d(
                x=[P[x, 0]], y=[P[x, 1]], z=[P[x, 2]], mode="markers+text", text=[letters[x]],
                textposition="top center", showlegend=False,
                marker=dict(size=8, color=ARM_COLORS[x], line=dict(color="black", width=2))),
                row=1, col=col)
        # edges: connect vertex i → edge points s..e → vertex j
        for k, (i, j, s, e, pk) in enumerate(edge_specs):
            seq = [i] + list(range(s, e + 1)) + [j]
            E = P[seq]
            fig.add_trace(go.Scatter3d(
                x=E[:, 0], y=E[:, 1], z=E[:, 2], mode="lines",
                line=dict(color=EDGE_COLORS[k % len(EDGE_COLORS)], width=4),
                name=f"{letters[i]}{letters[j]}", legendgroup=f"{letters[i]}{letters[j]}",
                showlegend=(col == 1)), row=1, col=col)

    fig.update_layout(
        title="MDS-of-distances: does activation geometry recover the behavior tetrahedron? "
              f"(L{d['layer']})  —  manifold r={d['r_manifold']:.2f}, linear r={d['r_linear']:.2f}",
        margin=dict(l=0, r=0, t=70, b=0), height=520, width=1300, legend=dict(x=1.0, y=0.5))
    for sc in ("scene", "scene2", "scene3"):
        fig.update_layout(**{sc: dict(xaxis_title="", yaxis_title="", zaxis_title="",
                                      xaxis_showticklabels=False, yaxis_showticklabels=False,
                                      zaxis_showticklabels=False)})
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"isometry_mds_L{d['layer']}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"manifold r={d['r_manifold']:.3f}  linear r={d['r_linear']:.3f}  (n={ref.shape[0]} nodes)")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
