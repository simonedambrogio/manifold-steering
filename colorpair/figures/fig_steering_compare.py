"""Manifold vs linear steering in ONE 2x2 figure (the headline comparison).

  Top row    : MANIFOLD steering (geodesic along the fitted spline)  + P(action) for the midpoint color.
  Bottom row : LINEAR steering (straight chord through activation space) + P(action) for the same color.
  Left column: the 3D-PCA ring with the steering path.   Right column: the P(action) curve (rainbow by position).

Both rows animate together on Play (paths grow; curves draw in left-to-right). Manifold -> a clean bump as
the steer passes the color; linear -> flat (the chord teleports past it). Static PDF shows the finished state.

    PYTHONPATH=. .venv/bin/python colorpair/figures/fig_steering_compare.py \
        --out colorpair/figures/colorpair_steering_compare
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

GEO_COLOR, LIN_COLOR = "#111111", "#c0392b"


def steer_path(phi, cs, mode, A, B, off, K):
    """Return (waypoints[K,k] for the readout, dense path[600,k] for drawing)."""
    if mode == "geodesic":
        return arclen_resample(cs, A, off, K), cs(np.linspace(A, A + off, 600))
    ts = np.linspace(0.0, 1.0, K)[:, None]
    tsd = np.linspace(0.0, 1.0, 600)[:, None]
    return (1 - ts) * phi[A] + ts * phi[B], (1 - tsd) * phi[A] + tsd * phi[B]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("colorpair/figures/colorpair_steering_compare"))
    ap.add_argument("--rows", choices=["both", "top"], default="both",
                    help="both = manifold(top)+linear(bottom); top = only the manifold row (bottom left blank for a staged reveal)")
    ap.add_argument("--start", type=int, default=4); ap.add_argument("--end", type=int, default=16)
    ap.add_argument("--targets", type=str, default="10",
                    help="comma-separated colors to plot P(action) for (one line each; the median is the labelled midpoint)")
    ap.add_argument("--k-way", type=int, default=73)
    ap.add_argument("--start-label", type=str, default="yellow")
    ap.add_argument("--end-label", type=str, default="blue")
    ap.add_argument("--target-label", type=str, default="green")
    args = ap.parse_args()
    sl, el, tl = args.start_label, args.end_label, args.target_label
    targets = [int(t) for t in args.targets.split(",")]
    multi = len(targets) > 1

    ckpt = torch.load(args.checkpoint, weights_only=False)
    n = int(ckpt["config"]["n_colors"]); k = int(ckpt["config"]["k"])
    model = ColorPairNet(n, k=k); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        phi = model.embed_all().numpy().astype(np.float64)
    phi_t = torch.tensor(phi, dtype=torch.float32)

    A, B, K = args.start, args.end, args.k_way
    off = (B - A) % n or n
    mid = targets[len(targets) // 2]                              # midpoint color (labelled on the x-axis)
    mid_frac = ((mid - A) % n) / off
    cs = periodic_spline(phi)

    # shared projection + ring geometry
    mu, Vt3 = pca_basis(phi)
    coords = (phi - mu) @ Vt3.T
    center = coords.mean(0)
    scale = max(np.ptp(coords[:, 0] - center[0]), np.ptp(coords[:, 1] - center[1])) + 1e-9
    proj = lambda X: (((X - mu) @ Vt3.T) - center) / scale
    ring = proj(phi)
    full3 = proj(cs(np.linspace(0, n, 800)))
    aspect = np.ptp(ring, axis=0)
    z_floor = ring[:, 2].min() - 0.35 * (np.ptp(ring[:, 2]) + 1e-9)
    cols = [matplotlib.colors.to_hex(CYC(i / n)) for i in range(n)]
    progress = np.linspace(0, 1, K)
    ptitle = "P(action) per color" if multi else f"P(action) for {tl}"

    # per-row steering paths + readouts (only the manifold row when --rows top); one P-curve per target color
    modes = [("geodesic", GEO_COLOR)] if args.rows == "top" else [("geodesic", GEO_COLOR), ("linear", LIN_COLOR)]
    rows = []
    for mode, pcolor in modes:
        wp, path_pts = steer_path(phi, cs, mode, A, B, off, K)
        wp_t = [torch.tensor(wp[i], dtype=torch.float32) for i in range(K)]
        ps = [np.array([identification_prob(model, phi_t, wp_t[i], c, n) for i in range(K)]) for c in targets]
        rows.append(dict(mode=mode, color=pcolor, arc3=proj(path_pts), ps=ps,
                         title=("Manifold" if mode == "geodesic" else "Linear")))

    bottom_titles = ("", "") if args.rows == "top" else (f"Linear steering: {sl} &#8594; {el}", ptitle)
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.06, vertical_spacing=0.10,
        specs=[[{"type": "scene"}, {"type": "xy"}], [{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(f"Manifold steering: {sl} &#8594; {el}", ptitle, *bottom_titles))

    def add_static_ring(r):                                        # ring + floor + spline + spheres + start/end/target
        fig.add_trace(go.Scatter3d(x=full3[:, 0], y=full3[:, 1], z=[z_floor] * len(full3), mode="lines",
                                   line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
                                   hoverinfo="skip", showlegend=False), r, 1)
        fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=[z_floor] * n, mode="markers",
                                   marker=dict(size=8, color=SHADOW_COLOR, symbol="circle"), opacity=SHADOW_OPACITY,
                                   hoverinfo="skip", showlegend=False), r, 1)
        fig.add_trace(go.Scatter3d(x=full3[:, 0], y=full3[:, 1], z=full3[:, 2], mode="lines",
                                   line=dict(color=EDGE_COLOR, width=4), hoverinfo="skip", showlegend=False), r, 1)
        fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="markers",
                                   marker=dict(size=11, color=list(range(n)), colorscale=hsv_colorscale(),
                                               cmin=0, cmax=n, symbol="circle", line=dict(width=1, color="black")),
                                   hoverinfo="skip", showlegend=False), r, 1)
        marks = {}
        for idx, lab in [(A, sl), (B, el), (mid, tl)]:
            marks.setdefault(idx, lab)
        for idx, lab in marks.items():
            fig.add_trace(go.Scatter3d(x=[ring[idx, 0]], y=[ring[idx, 1]], z=[ring[idx, 2]], mode="markers+text",
                                       marker=dict(size=15, color=cols[idx], symbol="circle", line=dict(width=2, color="black")),
                                       text=[lab], textposition="top center", textfont=dict(size=14),
                                       hoverinfo="skip", showlegend=False), r, 1)

    # Build rows. The newly-revealed steer ANIMATES (linear when both rows show; the manifold when it's
    # the only row); any earlier row (the manifold, in the 2x2) is drawn STATIC and COMPLETE so Play
    # only drives the new one.
    for ri, row in enumerate(rows, start=1):
        add_static_ring(ri)
        arc3 = row["arc3"]
        an = (args.rows == "top") or (row["mode"] == "linear")
        row["animated"] = an
        row["PATH_I"] = len(fig.data)
        px = (arc3[:1, 0], arc3[:1, 1], arc3[:1, 2]) if an else (arc3[:, 0], arc3[:, 1], arc3[:, 2])
        fig.add_trace(go.Scatter3d(x=px[0], y=px[1], z=px[2], mode="lines",
                                   line=dict(color=row["color"], width=7), hoverinfo="skip", showlegend=False), ri, 1)
        row["DOT_I"] = len(fig.data)                              # head marker (only for the animated row)
        hd = ([arc3[0, 0]], [arc3[0, 1]], [arc3[0, 2]]) if an else ([], [], [])
        fig.add_trace(go.Scatter3d(x=hd[0], y=hd[1], z=hd[2], mode="markers",
                                   marker=dict(size=4, color=row["color"], symbol="circle"),
                                   hoverinfo="skip", showlegend=False), ri, 1)
        row["CURVE_IS"] = []                                      # one light line per target color
        for c, pc in zip(targets, row["ps"]):
            row["CURVE_IS"].append(len(fig.data))
            cx, cy = ([], []) if an else (progress, pc)
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=cols[c], width=4),
                                     hoverinfo="skip", showlegend=False), ri, 2)

    # ---------- frames: grow both paths + reveal both curves together ----------
    nd = len(rows[0]["arc3"])
    animated_rows = [r for r in rows if r["animated"]]
    static_rows = [r for r in rows if not r["animated"]]         # e.g. the completed manifold row in the 2x2

    def static_curve_updates():
        """Repaint every static row's 2D curves at FULL each frame, else Plotly drops them during playback."""
        tr, dt = [], []
        for r in static_rows:
            for ci, pc in zip(r["CURVE_IS"], r["ps"]):
                tr.append(ci); dt.append(go.Scatter(x=progress, y=pc))
        return tr, dt

    # frame 0: animated rows reset to a point at the start; static curves stay full
    f0_traces, f0_data = [], []
    for r in animated_rows:
        f0_traces += [r["PATH_I"], r["DOT_I"]] + r["CURVE_IS"]
        f0_data += [go.Scatter3d(x=r["arc3"][:1, 0], y=r["arc3"][:1, 1], z=r["arc3"][:1, 2]),
                    go.Scatter3d(x=[r["arc3"][0, 0]], y=[r["arc3"][0, 1]], z=[r["arc3"][0, 2]])]
        f0_data += [go.Scatter(x=[], y=[]) for _ in r["CURVE_IS"]]
    st_traces, st_data = static_curve_updates()
    frames = [go.Frame(name="0", traces=f0_traces + st_traces, data=f0_data + st_data)]
    for f in range(1, K):
        m = max(1, int(round(f / (K - 1) * (nd - 1))) + 1)
        traces, data = [], []
        for r in animated_rows:
            traces += [r["PATH_I"], r["DOT_I"]] + r["CURVE_IS"]
            a = r["arc3"]
            data += [go.Scatter3d(x=a[:m, 0], y=a[:m, 1], z=a[:m, 2]),
                     go.Scatter3d(x=[a[m - 1, 0]], y=[a[m - 1, 1]], z=[a[m - 1, 2]])]
            data += [go.Scatter(x=progress[:f + 1], y=pc[:f + 1]) for pc in r["ps"]]
        traces += st_traces; data += st_data                    # keep static curves painted every frame
        frames.append(go.Frame(name=str(f), traces=traces, data=data))
    fig.frames = frames
    # the 2x2 animates only the linear steer -> run it 2x faster (8ms); the manifold-only step stays 16ms
    dur = 8 if args.rows == "both" else 16
    play_args = dict(frame=dict(duration=dur, redraw=True), fromcurrent=False,
                     transition=dict(duration=0), mode="immediate")

    axis3d = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                  showticklabels=False, title="")
    scene = dict(xaxis=axis3d, yaxis=axis3d, zaxis={**axis3d, "backgroundcolor": FLOOR_COLOR},
                 xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=True,
                 aspectmode="manual", aspectratio=dict(x=float(aspect[0]), y=float(aspect[1]), z=float(aspect[2])),
                 camera=dict(eye=dict(x=1.15, y=1.15, z=0.85)))
    # when only the top row is drawn, blank the bottom 3D scene so it's empty white (room for the reveal)
    blank3d = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="white")
    fig.update_layout(
        scene=scene, scene2=(blank3d if args.rows == "top" else scene),
        font=dict(size=16), paper_bgcolor="white", plot_bgcolor="white",
        width=1180, height=1060, margin=dict(l=0, r=20, t=50, b=60),
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.04, xanchor="left", y=-0.01, yanchor="top", pad=dict(t=0, r=8),
                          buttons=[
                              dict(label="&#9654;  Play", method="animate", args=[None, play_args]),
                              dict(label="&#10073;&#10073;  Pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                          ])])
    drawn_rows = [1] if args.rows == "top" else [1, 2]
    for r in drawn_rows:
        fig.update_xaxes(title_text=f"{sl} &#8594; {el}", range=[0, 1], tickvals=[0, mid_frac, 1],
                         ticktext=[sl, tl, el], showgrid=False, linecolor="black", row=r, col=2)
        fig.update_yaxes(title_text="P(action)", range=[-0.05, 1], tickvals=[0, 0.5, 1],
                         showgrid=True, gridcolor="#eee", linecolor="black", row=r, col=2)
    if args.rows == "top":                                        # hide the empty bottom-right xy panel
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = args.out.with_suffix(".html"); fig.write_html(str(html), auto_play=False,
        config={"responsive": False, "toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"saved -> {html}")
    # static PDF = finished figure: fill animated traces to full, drop controls
    for r in rows:
        fig.data[r["PATH_I"]].update(x=r["arc3"][:, 0], y=r["arc3"][:, 1], z=r["arc3"][:, 2])
        fig.data[r["DOT_I"]].update(x=[], y=[], z=[])
        for ci, pc in zip(r["CURVE_IS"], r["ps"]):
            fig.data[ci].update(x=progress, y=pc)
    fig.layout.updatemenus = None; fig.layout.sliders = None
    try:
        fig.write_image(str(args.out.with_suffix(".pdf")), scale=3)
        print(f"saved -> {args.out.with_suffix('.pdf')}")
    except Exception as e:
        print(f"(static export skipped: {e})")


if __name__ == "__main__":
    main()
