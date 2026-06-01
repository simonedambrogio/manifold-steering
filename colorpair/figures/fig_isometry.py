"""Figure 3 (color-pair task): scaled isometry M_h <-> M_y, in the manifold-steering paper's style.

Replicates the right portion of the paper's panel (a):
  (top row)    Activation ring M_h (cyclic diamonds, grey floor) with the two steering paths overlaid:
               the MANIFOLD path (geodesic, hugs the ring) and the LINEAR path (chord, cuts the interior),
               coloured by progress A -> antipodal B.
  (bottom row) Scaled-isometry scatters: behavior arc length (D_Y) vs activation distance (D_X) measured
               the manifold way (r high) and the linear way (r lower). Same D_Y; two metrics on M_h.

Inputs from `analyses.colorpair_isometry`. Run under the plotly interpreter, from the repo root:

    PYTHONPATH=. .venv/bin/python colorpair/figures/fig_isometry.py \
        --data artifacts/manifolds/colorpair_isometry.pt --out colorpair/figures/colorpair_isometry
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CYC = matplotlib.colormaps["hsv"]
FLOOR_COLOR = "#e5e5e5"
EDGE_COLOR = "rgba(100,100,100,0.45)"
SHADOW_COLOR, SHADOW_OPACITY = "#666666", 0.15
SCATTER_BLUE, FIT_RED = "steelblue", "#c0392b"


def plasma_scale(steps=64):
    cm = matplotlib.colormaps["plasma"]
    return [[i / (steps - 1), matplotlib.colors.to_hex(cm(i / (steps - 1)))] for i in range(steps)]


def add_ring_with_path(fig, ring, path, n, row, col):
    """Add the cyclic M_h ring + a progress-coloured steering path to a 3D subplot."""
    cols = [matplotlib.colors.to_hex(CYC(i / n)) for i in range(n)]
    closed = np.vstack([ring, ring[:1]])
    zr = ring[:, 2].ptp()
    xy_scale = max(ring[:, 0].ptp(), ring[:, 1].ptp())
    z_floor = ring[:, 2].min() - (0.35 * zr if zr > 1e-6 else 0.22 * xy_scale)  # flat ring: floor below
    # explicit floor mesh (reliable for flat rings, unlike the auto z-axis background)
    pad = 0.18 * xy_scale
    xs = [ring[:, 0].min() - pad, ring[:, 0].max() + pad]
    ys = [ring[:, 1].min() - pad, ring[:, 1].max() + pad]
    fig.add_trace(go.Mesh3d(x=[xs[0], xs[1], xs[1], xs[0]], y=[ys[0], ys[0], ys[1], ys[1]],
                            z=[z_floor] * 4, i=[0, 0], j=[1, 2], k=[2, 3], color=FLOOR_COLOR,
                            opacity=0.5, hoverinfo="skip", showlegend=False), row=row, col=col)
    # faint ring footprint on the floor
    fig.add_trace(go.Scatter3d(x=closed[:, 0], y=closed[:, 1], z=[z_floor] * len(closed), mode="lines",
                               line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
                               hoverinfo="skip", showlegend=False), row=row, col=col)
    # ring
    fig.add_trace(go.Scatter3d(x=closed[:, 0], y=closed[:, 1], z=closed[:, 2], mode="lines",
                               line=dict(color=EDGE_COLOR, width=4), hoverinfo="skip",
                               showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="markers",
                               marker=dict(size=5, color=list(range(n)),
                                           colorscale=[[i / (n - 1), c] for i, c in enumerate(cols)],
                                           symbol="diamond", line=dict(width=0.5, color="black")),
                               hoverinfo="skip", showlegend=False), row=row, col=col)
    # steering path, coloured by progress
    prog = np.linspace(0, 1, len(path))
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines+markers",
                               line=dict(color="#333", width=3),
                               marker=dict(size=4, color=prog, colorscale=plasma_scale(), showscale=False),
                               hoverinfo="skip", showlegend=False), row=row, col=col)


def add_isometry_scatter(fig, dx, dy, r, row, col):
    fig.add_trace(go.Scatter(x=dx, y=dy, mode="markers",
                             marker=dict(size=4, color=SCATTER_BLUE, opacity=0.3, line=dict(width=0)),
                             hoverinfo="skip", showlegend=False), row=row, col=col)
    slope = float(np.sum(dx * dy) / np.sum(dx * dx))
    xr = np.linspace(0, dx.max() * 1.02, 50)
    fig.add_trace(go.Scatter(x=xr, y=slope * xr, mode="lines", line=dict(color=FIT_RED, width=2),
                             hoverinfo="skip", showlegend=False), row=row, col=col)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("artifacts/manifolds/colorpair_isometry.pt"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("colorpair/figures/colorpair_isometry"))
    args = ap.parse_args()

    d = torch.load(args.data, weights_only=False)
    n = int(d["n_colors"])
    man_ring, man_path = d["man_ring"].numpy(), d["man_path_mds"].numpy()   # MDS(geodesic) embedding
    lin_ring, lin_path = d["lin_ring"].numpy(), d["lin_path_mds"].numpy()   # MDS(chord) embedding
    sc = d["scatter"]; dx_man, dx_lin, dy = sc["dx_man"].numpy(), sc["dx_lin"].numpy(), sc["dy"].numpy()
    r_man, r_lin = d["r_manifold"], d["r_linear"]
    print(f"manifold r={r_man:.3f}  linear r={r_lin:.3f}")

    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.08, horizontal_spacing=0.12,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("MDS of geodesic matrix — Manifold path", "MDS of chord matrix — Linear path",
                        f"Manifold paths    r = {r_man:.2f}", f"Linear paths    r = {r_lin:.2f}"))

    add_ring_with_path(fig, man_ring, man_path, n, 1, 1)
    add_ring_with_path(fig, lin_ring, lin_path, n, 1, 2)
    add_isometry_scatter(fig, dx_man, dy, r_man, 2, 1)
    add_isometry_scatter(fig, dx_lin, dy, r_lin, 2, 2)

    axis3d = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                  showticklabels=False, title="")
    scene = dict(xaxis=axis3d, yaxis=axis3d, zaxis=axis3d,
                 xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
                 aspectmode="data", camera=dict(eye=dict(x=1.0, y=1.0, z=1.7)))
    fig.update_layout(scene=scene, scene2=scene, paper_bgcolor="white",
                      width=1000, height=920, margin=dict(l=10, r=10, b=10, t=60),
                      title=dict(text="Color-pair scaled isometry: activation manifold vs behavior",
                                 x=0.5, xanchor="center"))
    for ax in ("xaxis3", "yaxis3", "xaxis4", "yaxis4"):
        fig.update_layout(**{ax: dict(showgrid=False, zeroline=False, rangemode="tozero",
                                      linecolor="black", ticks="outside")})
    fig.update_xaxes(title_text="Activation path length", row=2, col=1)
    fig.update_xaxes(title_text="Activation path length", row=2, col=2)
    fig.update_yaxes(title_text="Behavior arc length", row=2, col=1)
    fig.update_yaxes(title_text="Behavior arc length", row=2, col=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = args.out.with_suffix(".html"); fig.write_html(str(html),
        config={"toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"saved -> {html}")
    try:
        pdf = args.out.with_suffix(".pdf"); fig.write_image(str(pdf), scale=3)
        print(f"saved -> {pdf}")
    except Exception as e:
        print(f"(static export skipped: {e})")


if __name__ == "__main__":
    main()
