"""Figure 1 (color-pair task): the learned color manifold in 3D PCA space.

Mirrors the manifold-steering paper's structural figures (e.g. the weekday ring on a grey floor):
each color's encoder embedding phi(color) in R^k is projected to its top-3 principal components and
drawn as a diamond, coloured cyclically by ring position, with a closed polyline through the colors in
ring order, a grey floor plane and floor shadows. If the network internalized the latent as a ring, the
colors lie on a closed loop and the seam color (N-1) sits next to color 0.

Self-contained (no causalab / colorpair imports) so it can run under any interpreter that has
torch + plotly (+ kaleido for static export):

    .venv/bin/python colorpair/figures/fig_pca3d.py \
        --checkpoint artifacts/agent/colorpair_optB.pt --out colorpair/figures/colorpair_pca3d
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib
import plotly.graph_objects as go

CYC = matplotlib.colormaps["hsv"]                  # bright cyclic palette (paper's rainbow)
FLOOR_COLOR = "#e5e5e5"
EDGE_COLOR = "rgba(100,100,100,0.45)"
SHADOW_COLOR = "#666666"
SHADOW_OPACITY = 0.15


def hsv_colorscale(steps: int = 256):
    return [[i / (steps - 1), matplotlib.colors.to_hex(CYC(i / (steps - 1)))] for i in range(steps)]


def pca_3d(phi: np.ndarray):
    mu = phi.mean(0)
    U, S, Vt = np.linalg.svd(phi - mu, full_matrices=False)
    coords = (phi - mu) @ Vt[:3].T
    var = (S[:3] ** 2) / (S ** 2).sum()
    return coords, var


def build_figure(coords: np.ndarray, n: int, title: str, eye=(1.15, 1.15, 0.85)) -> go.Figure:
    # We want BOTH a matched floor-plane zoom across figures AND fully honest geometry.
    # Pure aspectmode="data" can't do both (it fits the whole box, so a taller/bowed
    # manifold shrinks). Instead: normalize the SCALE only (divide by the larger of the
    # x/y extents, so max(x,y) == 1 for every figure -> matched zoom), then take all three
    # aspect-ratio axes FROM THE DATA. Every axis is honest: a flatter manifold renders
    # flatter, an x-elongated one renders elongated; only the overall size is equalized.
    cols = [matplotlib.colors.to_hex(CYC(i / n)) for i in range(n)]
    coords = np.asarray(coords, dtype=float)
    coords = coords - coords.mean(0)
    coords = coords / (max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) + 1e-9)
    aspect = np.ptp(coords, axis=0)                                # true x:y:z ratios (max(x,y) == 1)
    closed = np.vstack([coords, coords[:1]])                       # ring order, loop closed
    z_floor = coords[:, 2].min() - 0.35 * (np.ptp(coords[:, 2]) + 1e-9)

    fig = go.Figure()
    # floor shadows (ring line + markers projected down)
    fig.add_trace(go.Scatter3d(
        x=closed[:, 0], y=closed[:, 1], z=[z_floor] * len(closed), mode="lines",
        line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=[z_floor] * n, mode="markers",
        marker=dict(size=9, color=SHADOW_COLOR, symbol="circle"), opacity=SHADOW_OPACITY,
        hoverinfo="skip", showlegend=False))
    # the ring polyline through the colors
    fig.add_trace(go.Scatter3d(
        x=closed[:, 0], y=closed[:, 1], z=closed[:, 2], mode="lines",
        line=dict(color=EDGE_COLOR, width=4), hoverinfo="skip", showlegend=False))
    # the colors themselves (diamonds), cyclically coloured + a cyclic colorbar
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
        marker=dict(size=13, color=list(range(n)), colorscale=hsv_colorscale(), cmin=0, cmax=n,
                    symbol="circle", line=dict(width=1, color="black"),
                    colorbar=dict(title="color identity<br>(ring position)", x=1.02, thickness=18,
                                  tickvals=[0, n // 4, n // 2, 3 * n // 4, n - 1])),
        text=[f"color {i}" for i in range(n)], hovertemplate="%{text}<extra></extra>",
        showlegend=False))

    axis = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                showticklabels=False, title="")
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=26)),
        font=dict(size=17),                                        # global text (colorbar label + ticks)
        scene=dict(xaxis=axis, yaxis=axis, zaxis={**axis, "backgroundcolor": FLOOR_COLOR},
                   xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=True,
                   aspectmode="manual",
                   aspectratio=dict(x=float(aspect[0]), y=float(aspect[1]), z=float(aspect[2])),
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
        paper_bgcolor="white", margin=dict(l=0, r=80, b=0, t=50), width=820, height=720)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("colorpair/figures/colorpair_pca3d"),
                    help="output path stem (.html and .pdf are written)")
    ap.add_argument("--title", type=str, default="Activation manifold")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    phi = ckpt["phi"].numpy().astype(np.float64)
    n = int(ckpt["config"]["n_colors"])
    coords, var = pca_3d(phi)
    print(f"N={n}, k={phi.shape[1]}; PCA var top-3 = "
          f"{var[0]:.3f}, {var[1]:.3f}, {var[2]:.3f} (sum {var.sum():.3f})")

    fig = build_figure(coords, n, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = args.out.with_suffix(".html"); fig.write_html(str(html),
        config={"responsive": False, "toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"saved -> {html}")
    try:
        pdf = args.out.with_suffix(".pdf"); fig.write_image(str(pdf), scale=3)
        print(f"saved -> {pdf}")
    except Exception as e:
        print(f"(static export skipped: {e})")


if __name__ == "__main__":
    main()
