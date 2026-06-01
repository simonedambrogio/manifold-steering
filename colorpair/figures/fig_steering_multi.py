"""Steering with MANY P(action) curves -- the colorpair analog of the paper's P(token) panel.

Same two-panel layout and animation as fig_manifold_steering, but the right panel plots P(action)
for several evenly-spaced colors along the steer (default: yellow, a yellow-green, the green midpoint,
a cyan, blue), each line drawn IN THE COLOR IT REPRESENTS (solid, not a position gradient).

  - --mode geodesic : steer ALONG the fitted manifold -> the curves are ordered bumps that march
                      across as the steer passes each color (behavior tracks the represented color).
  - --mode linear   : straight chord phi(start) -> phi(end) -> the intermediate curves stay ~0
                      (the chord teleports through off-manifold space, skipping those colors).

    PYTHONPATH=. .venv/bin/python colorpair/figures/fig_steering_multi.py --mode geodesic \
        --out colorpair/figures/colorpair_manifold_steering_multi
    PYTHONPATH=. .venv/bin/python colorpair/figures/fig_steering_multi.py --mode linear \
        --out colorpair/figures/colorpair_linear_steering_multi
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from colorpair.model import ColorPairNet
from colorpair.figures.fig_pca3d import (
    CYC, FLOOR_COLOR, EDGE_COLOR, SHADOW_COLOR, SHADOW_OPACITY, hsv_colorscale,
)
from colorpair.figures.fig_manifold_steering import (
    identification_prob, periodic_spline, arclen_resample, pca_basis,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("colorpair/figures/colorpair_manifold_steering_multi"))
    ap.add_argument("--mode", choices=["geodesic", "linear"], default="geodesic")
    ap.add_argument("--start", type=int, default=4, help="start color (yellow)")
    ap.add_argument("--end", type=int, default=16, help="end color (blue)")
    ap.add_argument("--targets", type=str, default="4,7,10,13,16",
                    help="comma-separated colors to plot P(action) for (evenly spaced start..end)")
    ap.add_argument("--k-way", type=int, default=73, help="waypoints along the steer")
    ap.add_argument("--start-label", type=str, default="yellow")
    ap.add_argument("--end-label", type=str, default="blue")
    ap.add_argument("--mid-label", type=str, default="green")
    args = ap.parse_args()
    start_label, end_label, mid_label = args.start_label, args.end_label, args.mid_label
    targets = [int(t) for t in args.targets.split(",")]

    ckpt = torch.load(args.checkpoint, weights_only=False)
    n = int(ckpt["config"]["n_colors"]); k = int(ckpt["config"]["k"])
    model = ColorPairNet(n, k=k); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        phi = model.embed_all().numpy().astype(np.float64)
    phi_t = torch.tensor(phi, dtype=torch.float32)

    A, B, K = args.start, args.end, args.k_way
    off = (B - A) % n or n
    mid = targets[len(targets) // 2]                                          # the "current intermediate color"
    mid_frac = ((mid - A) % n) / off

    # --- steering path (geodesic ALONG the manifold, or a straight linear chord) ---
    cs = periodic_spline(phi)
    if args.mode == "geodesic":
        wp = arclen_resample(cs, A, off, K)
        path_pts = cs(np.linspace(A, A + off, 600))
        path_color, mode_word = "#111111", "Manifold"
    else:
        ts = np.linspace(0.0, 1.0, K)[:, None]
        wp = (1 - ts) * phi[A] + ts * phi[B]
        tsd = np.linspace(0.0, 1.0, 600)[:, None]
        path_pts = (1 - tsd) * phi[A] + tsd * phi[B]
        path_color, mode_word = "#c0392b", "Linear"

    # P(action) for each target color along the steer (choices only)
    wp_t = [torch.tensor(wp[i], dtype=torch.float32) for i in range(K)]
    p_curves = [np.array([identification_prob(model, phi_t, wp_t[i], c, n) for i in range(K)]) for c in targets]
    progress = np.linspace(0, 1, K)

    # --- project to the shared, normalized PCA space (mirrors fig_pca3d) ---
    mu, Vt3 = pca_basis(phi)
    coords = (phi - mu) @ Vt3.T
    center = coords.mean(0)
    scale = max(np.ptp(coords[:, 0] - center[0]), np.ptp(coords[:, 1] - center[1])) + 1e-9
    proj = lambda X: (((X - mu) @ Vt3.T) - center) / scale
    ring = proj(phi)
    full3 = proj(cs(np.linspace(0, n, 800)))
    arc3 = proj(path_pts)
    aspect = np.ptp(ring, axis=0)
    z_floor = ring[:, 2].min() - 0.35 * (np.ptp(ring[:, 2]) + 1e-9)
    cols = [matplotlib.colors.to_hex(CYC(i / n)) for i in range(n)]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.06,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(f"{mode_word} steering: {start_label} &#8594; {end_label}", "P(action) per color"))

    # ---------- static traces (3D ring + floor + spline + spheres + start/end/mid markers) ----------
    fig.add_trace(go.Scatter3d(x=full3[:, 0], y=full3[:, 1], z=[z_floor] * len(full3), mode="lines",
                               line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
                               hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=[z_floor] * n, mode="markers",
                               marker=dict(size=8, color=SHADOW_COLOR, symbol="circle"), opacity=SHADOW_OPACITY,
                               hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter3d(x=full3[:, 0], y=full3[:, 1], z=full3[:, 2], mode="lines",
                               line=dict(color=EDGE_COLOR, width=4), hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="markers",
                               marker=dict(size=11, color=list(range(n)), colorscale=hsv_colorscale(),
                                           cmin=0, cmax=n, symbol="circle", line=dict(width=1, color="black")),
                               hoverinfo="skip", showlegend=False), 1, 1)
    marks = {}
    for idx, lab in [(A, start_label), (B, end_label), (mid, mid_label)]:
        marks.setdefault(idx, lab)
    for idx, lab in marks.items():
        fig.add_trace(go.Scatter3d(x=[ring[idx, 0]], y=[ring[idx, 1]], z=[ring[idx, 2]], mode="markers+text",
                                   marker=dict(size=16, color=cols[idx], symbol="circle", line=dict(width=2, color="black")),
                                   text=[lab], textposition="top center", textfont=dict(size=15),
                                   hoverinfo="skip", showlegend=False), 1, 1)

    # ---------- animated traces: path, moving head, then one solid line per target color ----------
    PATH_I = len(fig.data); DOT_I = PATH_I + 1; CURVE0 = PATH_I + 2
    fig.add_trace(go.Scatter3d(x=arc3[:1, 0], y=arc3[:1, 1], z=arc3[:1, 2], mode="lines",
                               line=dict(color=path_color, width=7), hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter3d(x=[arc3[0, 0]], y=[arc3[0, 1]], z=[arc3[0, 2]], mode="markers",
                               marker=dict(size=4, color=path_color, symbol="circle"),
                               hoverinfo="skip", showlegend=False), 1, 1)
    for c in targets:                                              # each line is drawn in the color it refers to
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line=dict(color=matplotlib.colors.to_hex(CYC(c / n)), width=4),
                                 hoverinfo="skip", showlegend=False), 1, 2)

    # ---------- frames: grow the path; draw all the P(action) curves in together ----------
    nd = len(arc3)
    curve_idx = list(range(CURVE0, CURVE0 + len(targets)))
    frames = [go.Frame(name="0", traces=[PATH_I, DOT_I] + curve_idx, data=(
        [go.Scatter3d(x=arc3[:1, 0], y=arc3[:1, 1], z=arc3[:1, 2]),
         go.Scatter3d(x=[arc3[0, 0]], y=[arc3[0, 1]], z=[arc3[0, 2]])]
        + [go.Scatter(x=[], y=[]) for _ in targets]))]
    for f in range(1, K):
        m = max(1, int(round(f / (K - 1) * (nd - 1))) + 1)
        kp = f + 1
        frames.append(go.Frame(name=str(f), traces=[PATH_I, DOT_I] + curve_idx, data=(
            [go.Scatter3d(x=arc3[:m, 0], y=arc3[:m, 1], z=arc3[:m, 2]),
             go.Scatter3d(x=[arc3[m - 1, 0]], y=[arc3[m - 1, 1]], z=[arc3[m - 1, 2]])]
            + [go.Scatter(x=progress[:kp], y=pc[:kp]) for pc in p_curves])))
    fig.frames = frames
    play_args = dict(frame=dict(duration=16, redraw=True), fromcurrent=False,
                     transition=dict(duration=0), mode="immediate")

    axis3d = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                  showticklabels=False, title="")
    fig.update_layout(
        scene=dict(xaxis=axis3d, yaxis=axis3d, zaxis={**axis3d, "backgroundcolor": FLOOR_COLOR},
                   xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=True,
                   aspectmode="manual",
                   aspectratio=dict(x=float(aspect[0]), y=float(aspect[1]), z=float(aspect[2])),
                   camera=dict(eye=dict(x=1.15, y=1.15, z=0.85))),
        font=dict(size=17), paper_bgcolor="white", plot_bgcolor="white",
        width=1280, height=620, margin=dict(l=0, r=20, t=55, b=70),
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.04, xanchor="left", y=-0.02, yanchor="top", pad=dict(t=0, r=8),
                          buttons=[
                              dict(label="&#9654;  Play", method="animate", args=[None, play_args]),
                              dict(label="&#10073;&#10073;  Pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                          ])])
    fig.update_xaxes(title_text=f"{start_label} &#8594; {end_label}", range=[0, 1],
                     tickvals=[0, mid_frac, 1], ticktext=[start_label, mid_label, end_label],
                     showgrid=False, linecolor="black", row=1, col=2)
    fig.update_yaxes(title_text="P(action)", range=[-0.05, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                     showgrid=True, gridcolor="#eee", linecolor="black", row=1, col=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = args.out.with_suffix(".html"); fig.write_html(str(html), auto_play=False,
        config={"responsive": False, "toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"saved -> {html}")
    # static PDF = the finished figure: fill animated traces to full, drop controls
    fig.data[PATH_I].update(x=arc3[:, 0], y=arc3[:, 1], z=arc3[:, 2])
    fig.data[DOT_I].update(x=[], y=[], z=[])
    for j, pc in enumerate(p_curves):
        fig.data[CURVE0 + j].update(x=progress, y=pc)
    fig.layout.updatemenus = None; fig.layout.sliders = None
    try:
        fig.write_image(str(args.out.with_suffix(".pdf")), scale=3)
        print(f"saved -> {args.out.with_suffix('.pdf')}")
    except Exception as e:
        print(f"(static export skipped: {e})")


if __name__ == "__main__":
    main()
