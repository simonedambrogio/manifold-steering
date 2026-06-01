"""Manifold steering on the color ring (first draft): a two-panel causal figure.

LEFT  : the 3D-PCA activation ring with the fitted MANIFOLD (a periodic cubic spline through the
        24 color embeddings) drawn on it. We steer one slot's embedding ALONG the spline -- so the
        path follows the smooth manifold, not straight chords between adjacent colors.
RIGHT : x = steering progress, y = P(action) for the target color, read from CHOICES ONLY.
        At each waypoint w we run two forced-choice trials -- is pair (w, target) closer than
        (target, target-1)? and than (target, target+1)? -- and average their soft choice
        probabilities. This is high only while w sits within ~2 ring steps of the target and ~0
        elsewhere: a localized, smooth bump (the colorpair analog of the paper's P(token)). The
        curve is colored by the steered ring position (yellow -> ... -> yellow).

    PYTHONPATH=. .venv/bin/python colorpair/figures/fig_manifold_steering.py \
        --checkpoint artifacts/agent/colorpair_optB.pt --out colorpair/figures/colorpair_manifold_steering
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

from colorpair.model import ColorPairNet
from colorpair.figures.fig_pca3d import (
    CYC, FLOOR_COLOR, EDGE_COLOR, SHADOW_COLOR, SHADOW_OPACITY, hsv_colorscale,
)

PATH_COLOR = "#111111"          # the manifold steering path (matches the paper's black "Manifold")


def periodic_spline(phi: np.ndarray) -> CubicSpline:
    """Periodic cubic spline through the color embeddings -- the fitted manifold M_h.

    Parameter t in [0, n] with cs(0) == cs(n). Evaluating at non-integer t moves smoothly BETWEEN
    colors ALONG the manifold (not along straight chords), which is the point: we steer on the spline."""
    n = len(phi)
    t = np.arange(n + 1)
    vals = np.vstack([phi, phi[:1]])                          # close the loop for bc_type='periodic'
    return CubicSpline(t, vals, bc_type="periodic")


def arclen_resample(cs: CubicSpline, A: int, off: int, K: int, n_dense: int = 4000) -> np.ndarray:
    """K embeddings ON the spline manifold, uniform in arc length, along the arc A -> A+off."""
    tt = np.linspace(A, A + off, n_dense)
    pts = cs(tt)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return cs(np.interp(np.linspace(0, cum[-1], K), cum, tt))


@torch.no_grad()
def identification_prob(model: ColorPairNet, phi_all: torch.Tensor, w: torch.Tensor, r: int, n: int) -> float:
    """Choices-only readout: how strongly the model judges the steered slot w to match color r.

    Two real forced-choice trials -- is pair (w, r) closer than (r, r-1)? and than (r, r+1)? -- averaged.
    Each is a soft choice probability sigmoid(g(r, r+-1) - g(w, r)). The distance-1 reference keeps it
    from saturating, so as the steer passes r this is a rounded, smooth bump that is ~0 elsewhere."""
    emb = torch.empty(2, 4, model.k)
    for j, rn in enumerate([(r - 1) % n, (r + 1) % n]):
        emb[j, 0] = w; emb[j, 1] = phi_all[r]; emb[j, 2] = phi_all[r]; emb[j, 3] = phi_all[rn]
    logit = model.logits_from_embeddings(emb)                # >0 => (w, r) is the closer pair
    return float(torch.sigmoid(logit).mean())


def pca_basis(phi: np.ndarray):
    """Return (mu, Vt3) so any embedding projects to the SAME 3D-PCA space as the ring."""
    mu = phi.mean(0)
    _, _, Vt = np.linalg.svd(phi - mu, full_matrices=False)
    return mu, Vt[:3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("artifacts/agent/colorpair_optB.pt"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("colorpair/figures/colorpair_manifold_steering"))
    ap.add_argument("--mode", choices=["geodesic", "linear"], default="geodesic",
                    help="geodesic = steer ALONG the fitted manifold; linear = straight chord through activation space")
    ap.add_argument("--start", type=int, default=4, help="start color")
    ap.add_argument("--end", type=int, default=16, help="end color (== start => full loop around the ring)")
    ap.add_argument("--target", type=int, default=10, help="color whose P(action) we plot (here: the midpoint)")
    ap.add_argument("--k-way", type=int, default=73, help="waypoints along the steer")
    ap.add_argument("--start-label", type=str, default="yellow")
    ap.add_argument("--end-label", type=str, default="blue")
    ap.add_argument("--target-label", type=str, default="green")
    args = ap.parse_args()
    start_label, end_label, target_label = args.start_label, args.end_label, args.target_label

    ckpt = torch.load(args.checkpoint, weights_only=False)
    n = int(ckpt["config"]["n_colors"]); k = int(ckpt["config"]["k"])
    model = ColorPairNet(n, k=k); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        phi = model.embed_all().numpy().astype(np.float64)
    phi_t = torch.tensor(phi, dtype=torch.float32)

    A, B, tgt, K = args.start, args.end, args.target, args.k_way
    off = (B - A) % n or n                                                    # start == end => full loop
    tgt_frac = ((tgt - A) % n) / off                                          # where the target sits along the steer

    # --- build the steering path (geodesic ALONG the manifold, or a straight linear chord) ---
    cs = periodic_spline(phi)                                                 # fitted manifold (drawn as context in both modes)
    if args.mode == "geodesic":
        wp = arclen_resample(cs, A, off, K)                                   # [K, k] ON the manifold
        path_pts = cs(np.linspace(A, A + off, 600))                           # dense path for the 3D curve
        path_color, mode_word = "#111111", "Manifold"
    else:                                                                     # linear: straight chord phi(A) -> phi(B)
        ts = np.linspace(0.0, 1.0, K)[:, None]
        wp = (1 - ts) * phi[A] + ts * phi[B]                                  # cuts through the ring interior (off-manifold)
        tsd = np.linspace(0.0, 1.0, 600)[:, None]
        path_pts = (1 - tsd) * phi[A] + tsd * phi[B]
        path_color, mode_word = "#c0392b", "Linear"
    p_target = np.array([identification_prob(model, phi_t, torch.tensor(wp[i], dtype=torch.float32), tgt, n)
                         for i in range(K)])
    progress = np.linspace(0, 1, K)

    # --- project everything into a common, normalized PCA space (mirrors fig_pca3d) ---
    mu, Vt3 = pca_basis(phi)
    coords = (phi - mu) @ Vt3.T
    center = coords.mean(0)
    scale = max(np.ptp(coords[:, 0] - center[0]), np.ptp(coords[:, 1] - center[1])) + 1e-9
    proj = lambda X: (((X - mu) @ Vt3.T) - center) / scale
    ring = proj(phi)                                                          # the 24 colors (lie on the spline)
    full3 = proj(cs(np.linspace(0, n, 800)))                                  # full manifold spline (context)
    arc3 = proj(path_pts)                                                     # the steered path (manifold arc or linear chord)
    aspect = np.ptp(ring, axis=0)
    z_floor = ring[:, 2].min() - 0.35 * (np.ptp(ring[:, 2]) + 1e-9)
    cols = [matplotlib.colors.to_hex(CYC(i / n)) for i in range(n)]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.06,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(f"{mode_word} steering: {start_label} &#8594; {end_label}", f"P(action) for {target_label}"))

    # ---------- static traces ----------
    fig.add_trace(go.Scatter3d(x=full3[:, 0], y=full3[:, 1], z=[z_floor] * len(full3), mode="lines",
                               line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
                               hoverinfo="skip", showlegend=False), 1, 1)                       # floor shadow line
    fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=[z_floor] * n, mode="markers",
                               marker=dict(size=8, color=SHADOW_COLOR, symbol="circle"), opacity=SHADOW_OPACITY,
                               hoverinfo="skip", showlegend=False), 1, 1)                       # floor shadow dots
    fig.add_trace(go.Scatter3d(x=full3[:, 0], y=full3[:, 1], z=full3[:, 2], mode="lines",
                               line=dict(color=EDGE_COLOR, width=4), hoverinfo="skip", showlegend=False), 1, 1)  # grey manifold
    fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="markers",
                               marker=dict(size=11, color=list(range(n)), colorscale=hsv_colorscale(),
                                           cmin=0, cmax=n, symbol="circle", line=dict(width=1, color="black")),
                               hoverinfo="skip", showlegend=False), 1, 1)                       # colored spheres
    marks = {}                                                    # start, end, target (dedup if any coincide)
    for idx, lab in [(A, start_label), (B, end_label), (tgt, target_label)]:
        marks.setdefault(idx, lab)
    for idx, lab in marks.items():
        fig.add_trace(go.Scatter3d(x=[ring[idx, 0]], y=[ring[idx, 1]], z=[ring[idx, 2]], mode="markers+text",
                                   marker=dict(size=16, color=cols[idx], symbol="circle", line=dict(width=2, color="black")),
                                   text=[lab], textposition="top center", textfont=dict(size=15),
                                   hoverinfo="skip", showlegend=False), 1, 1)                   # start / end / target

    # ---------- animated traces (base = START: a point at the start; Play grows it. PDF filled to full below) ----------
    PATH_I = len(fig.data); DOT_I = PATH_I + 1; SEG0 = PATH_I + 2  # indices AFTER the static traces (robust to marker count)
    fig.add_trace(go.Scatter3d(x=arc3[:1, 0], y=arc3[:1, 1], z=arc3[:1, 2], mode="lines",
                               line=dict(color=path_color, width=7), hoverinfo="skip", showlegend=False), 1, 1)  # growing path (a point at start)
    fig.add_trace(go.Scatter3d(x=[arc3[0, 0]], y=[arc3[0, 1]], z=[arc3[0, 2]], mode="markers",
                               marker=dict(size=4, color=path_color, symbol="circle"),
                               hoverinfo="skip", showlegend=False), 1, 1)                       # head at the start (yellow)
    seg_cols = [matplotlib.colors.to_hex(CYC(((A + off * 0.5 * (progress[i] + progress[i + 1])) % n) / n))
                for i in range(K - 1)]
    for i in range(K - 1):                                         # smooth rainbow curve = per-segment colored lines (hidden until Play)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line=dict(color=seg_cols[i], width=4), hoverinfo="skip",
                                 showlegend=False), 1, 2)

    # ---------- frames: grow the path around the ring + reveal the curve segments left-to-right ----------
    nd = len(arc3)
    frames = [go.Frame(name="0", traces=[PATH_I, DOT_I] + list(range(SEG0, SEG0 + K - 1)), data=(
        [go.Scatter3d(x=arc3[:1, 0], y=arc3[:1, 1], z=arc3[:1, 2]),                 # path = start point
         go.Scatter3d(x=[arc3[0, 0]], y=[arc3[0, 1]], z=[arc3[0, 2]])]              # head at start
        + [go.Scatter(x=[], y=[]) for _ in range(K - 1)]))]                         # hide every curve segment
    for f in range(1, K):                                          # grow path; reveal one more segment each step
        m = max(1, int(round(f / (K - 1) * (nd - 1))) + 1)
        frames.append(go.Frame(name=str(f), traces=[PATH_I, DOT_I, SEG0 + (f - 1)], data=[
            go.Scatter3d(x=arc3[:m, 0], y=arc3[:m, 1], z=arc3[:m, 2]),
            go.Scatter3d(x=[arc3[m - 1, 0]], y=[arc3[m - 1, 1]], z=[arc3[m - 1, 2]]),
            go.Scatter(x=progress[f - 1:f + 1], y=p_target[f - 1:f + 1]),
        ]))
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
                     tickvals=[0, tgt_frac, 1], ticktext=[start_label, target_label, end_label],
                     showgrid=False, linecolor="black", row=1, col=2)
    fig.update_yaxes(title_text="P(action)", range=[-0.05, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                     showgrid=True, gridcolor="#eee", linecolor="black", row=1, col=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = args.out.with_suffix(".html"); fig.write_html(str(html), auto_play=False,
        config={"responsive": False, "toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"saved -> {html}")
    # static PDF shows the FINISHED figure: fill the animated traces to full, then drop controls
    fig.data[PATH_I].update(x=arc3[:, 0], y=arc3[:, 1], z=arc3[:, 2])
    fig.data[DOT_I].update(x=[], y=[], z=[])
    for i in range(K - 1):
        fig.data[SEG0 + i].update(x=progress[i:i + 2], y=p_target[i:i + 2])
    fig.layout.updatemenus = None; fig.layout.sliders = None     # clean static export (no Play/Pause controls)
    try:
        fig.write_image(str(args.out.with_suffix(".pdf")), scale=3)
        print(f"saved -> {args.out.with_suffix('.pdf')}")
    except Exception as e:
        print(f"(static export skipped: {e})")


if __name__ == "__main__":
    main()
