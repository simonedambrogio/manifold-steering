"""Animated causal-steering figure (colorpair-style): steer along M_h, watch P(action).

Bandit analog of colorpair/figures/fig_steering_multi.py. LEFT: the floor-style raw-PCA
geometry (unsupervised, band 95–105) with the value-manifold tetrahedron; the chosen pair's
M_h edge is the steer path, which grows with a moving head. RIGHT (colorpair theme — white,
#eee grid): P(action) with ONE LINE PER ARM, each in its arm color, growing as the steer
sweeps the fraction t: 0 (arm i best) → 1 (arm j best).

    .venv/bin/python -m figures.steer_animation --pair 1_2 \
        --steering artifacts/steering/llama8b_fixed_pair_steering_L19_t95-105.pt \
        --trial-min 95 --trial-max 106 --outdir presentation/bandit/assets
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from figures.raw_pca3 import compute
from figures.value_relative_structure import ARM_COLORS, FLOOR_COLOR, SHADOW_COLOR, SHADOW_OPACITY


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steering", type=pathlib.Path,
                   default=pathlib.Path("artifacts/steering/llama8b_fixed_pair_steering_L19_t95-105.pt"))
    p.add_argument("--activations", type=pathlib.Path,
                   default=pathlib.Path("artifacts/activations/llama8b_fixed_residual.pt"))
    p.add_argument("--q", type=pathlib.Path, default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--trial-min", type=int, default=95)
    p.add_argument("--trial-max", type=int, default=106)
    p.add_argument("--n-show", type=int, default=4000)
    p.add_argument("--n-bins", type=int, default=12)
    p.add_argument("--pairs", type=pathlib.Path,
                   default=pathlib.Path("artifacts/manifolds/llama8b_fixed_pairs_L19_t95-105.pt"))
    p.add_argument("--pair", default="1_2", help="i_j arm indices (default C-D)")
    p.add_argument("--eye", type=float, nargs=3, default=[1.5, 1.5, 1.1])
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("presentation/bandit/assets"))
    args = p.parse_args()
    i, j = (int(x) for x in args.pair.split("_"))

    # ---- geometry (unsupervised raw top-3 PCA, band window), normalized like the floor figure ----
    d = compute(args)
    na, letters = d["na"], d["letters"]
    C = d["C"].astype(float); V = d["vertices"].astype(float)
    splines = {k: g.astype(float) for k, g in d["edges"].items()}
    ctr = C.mean(0)
    scale = max(np.ptp(C[:, 0] - ctr[0]), np.ptp(C[:, 1] - ctr[1])) + 1e-9
    norm = lambda P: (P - ctr) / scale
    C = norm(C); V = norm(V); splines = {k: norm(g) for k, g in splines.items()}
    aspect = np.ptp(C, axis=0); z_floor = C[:, 2].min() - 0.12 * (np.ptp(C[:, 2]) + 1e-9)
    arc3 = splines[(i, j)]; nd = len(arc3)                            # the steer path (C→D edge)

    # ---- induced P(action) along the manifold steer ----
    st = torch.load(args.steering, weights_only=False)
    pr = st["results"][args.pair]; K = st["K"]
    man = pr["dist_manifold"].numpy().mean(0)[:, :na]
    man = man / man.sum(1, keepdims=True)                            # [K,4]
    progress = np.linspace(0, 1, K)
    # natural (unintervened) M_y trajectory along the edge, for the grey reference
    pf = torch.load(args.pairs, weights_only=False)
    nat = pf["pairs"][args.pair]["dist_grid"].numpy()                # [n_grid,4]
    xn = np.linspace(0, 1, len(nat))
    lin = pr["dist_linear_insub"].numpy().mean(0)[:, :na]
    lin = lin / lin.sum(1, keepdims=True)                            # [K,4] linear-induced

    fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28], horizontal_spacing=0.06,
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=(f"Manifold steering: {letters[i]} &#8594; {letters[j]}",
                                        "P(action) per arm"))

    # ---------- static scene: floor shadows, faint cloud, non-focal edges, vertices ----------
    for g in splines.values():
        fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=[z_floor] * len(g), mode="lines",
            line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY, hoverinfo="skip",
            showlegend=False), 1, 1)
    fig.add_trace(go.Scatter3d(x=V[:, 0], y=V[:, 1], z=[z_floor] * na, mode="markers",
        marker=dict(size=8, color=SHADOW_COLOR), opacity=SHADOW_OPACITY, hoverinfo="skip",
        showlegend=False), 1, 1)
    for x in range(na):
        m = d["argmax_idx"] == x
        fig.add_trace(go.Scatter3d(x=C[m, 0], y=C[m, 1], z=C[m, 2], mode="markers", showlegend=False,
            marker=dict(size=2.2, color=ARM_COLORS[x], opacity=0.12), hoverinfo="skip"), 1, 1)
    for k, g in splines.items():
        if k == (i, j):
            continue
        fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="lines",
            line=dict(color="#cccccc", width=2), hoverinfo="skip", showlegend=False), 1, 1)
    for x in range(na):
        fig.add_trace(go.Scatter3d(x=[V[x, 0]], y=[V[x, 1]], z=[V[x, 2]], mode="markers+text",
            text=[letters[x]], textposition="top center", textfont=dict(size=16), showlegend=False,
            marker=dict(size=12, color=ARM_COLORS[x], line=dict(color="black", width=2))), 1, 1)
    # single 0.5 reference line (theme) — drawn first so it sits underneath
    fig.add_trace(go.Scatter(x=[0, 1], y=[0.5, 0.5], mode="lines", showlegend=False,
        line=dict(color="#e5e5e5", width=1), hoverinfo="skip"), 1, 2)
    # natural (unintervened) trajectory — grey reference, drawn under the induced lines
    for a in range(na):
        fig.add_trace(go.Scatter(x=xn, y=nat[:, a], mode="lines", showlegend=(a == 0),
            name="natural", legendgroup="natural", line=dict(color="#b0b0b0", width=2),
            hoverinfo="skip"), 1, 2)

    # ---------- animated: growing steer path + head dot, then one growing line per arm ----------
    PATH_I = len(fig.data); DOT_I = PATH_I + 1; CURVE0 = PATH_I + 2
    fig.add_trace(go.Scatter3d(x=arc3[:1, 0], y=arc3[:1, 1], z=arc3[:1, 2], mode="lines",
        line=dict(color="#111111", width=7), hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter3d(x=[arc3[0, 0]], y=[arc3[0, 1]], z=[arc3[0, 2]], mode="markers",
        marker=dict(size=7, color="black", line=dict(color="white", width=1.5)),
        hoverinfo="skip", showlegend=False), 1, 1)
    for a in range(na):
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=letters[a], showlegend=False,
            line=dict(color=ARM_COLORS[a], width=4)), 1, 2)
    curve_idx = list(range(CURVE0, CURVE0 + na))

    # ---------- linear steering (hidden by default; toggled by a button) ----------
    # 3D linear chord (solid red — plotly 3D lines can't be dotted) between the two vertices
    lin_path_idx = len(fig.data)
    chord = np.stack([V[i], V[j]])
    fig.add_trace(go.Scatter3d(x=chord[:, 0], y=chord[:, 1], z=chord[:, 2], mode="lines",
        line=dict(color="#c0392b", width=5), visible=False, hoverinfo="skip", showlegend=False), 1, 1)
    # right-panel linear-induced P(action) — dotted, per arm
    lin_curve_idx = []
    for a in range(na):
        fig.add_trace(go.Scatter(x=progress, y=lin[:, a], mode="lines", visible=False,
            name="linear" if a == 0 else None, legendgroup="linear", showlegend=(a == 0),
            line=dict(color=ARM_COLORS[a], width=3, dash="dot"), hoverinfo="skip"), 1, 2)
        lin_curve_idx.append(len(fig.data) - 1)
    lin_idx = [lin_path_idx] + lin_curve_idx

    frames = []
    for f in range(K):
        m = max(1, int(round(f / (K - 1) * (nd - 1))) + 1)
        kp = f + 1
        frames.append(go.Frame(name=str(f), traces=[PATH_I, DOT_I] + curve_idx, data=(
            [go.Scatter3d(x=arc3[:m, 0], y=arc3[:m, 1], z=arc3[:m, 2]),
             go.Scatter3d(x=[arc3[m - 1, 0]], y=[arc3[m - 1, 1]], z=[arc3[m - 1, 2]])]
            + [go.Scatter(x=progress[:kp], y=man[:kp, a]) for a in range(na)])))
    fig.frames = frames

    play_args = dict(frame=dict(duration=90, redraw=True), fromcurrent=True,
                     transition=dict(duration=0), mode="immediate")
    axis3d = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                  showticklabels=False, title="")
    fig.update_layout(
        scene=dict(xaxis=axis3d, yaxis=axis3d, zaxis={**axis3d, "backgroundcolor": FLOOR_COLOR},
                   xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=True,
                   aspectmode="manual",
                   aspectratio=dict(x=float(aspect[0]), y=float(aspect[1]), z=float(aspect[2])),
                   camera=dict(eye=dict(x=float(args.eye[0]), y=float(args.eye[1]), z=float(args.eye[2])))),
        font=dict(size=17), paper_bgcolor="white", plot_bgcolor="white",
        width=1280, height=620, margin=dict(l=0, r=20, t=55, b=80),
        legend=dict(x=0.99, y=0.98, xanchor="right", font=dict(size=14)),
        updatemenus=[
            dict(type="buttons", direction="left", showactive=False,
                 x=0.04, xanchor="left", y=-0.04, yanchor="top", pad=dict(t=0, r=8), buttons=[
                dict(label="&#9654;  Play", method="animate", args=[None, play_args]),
                dict(label="&#10073;&#10073;  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])]),
            dict(type="buttons", direction="left", showactive=True,
                 x=0.34, xanchor="left", y=-0.04, yanchor="top", pad=dict(t=0, r=8), buttons=[
                dict(label="linear: show", method="restyle", args=[{"visible": True}, lin_idx]),
                dict(label="linear: hide", method="restyle", args=[{"visible": False}, lin_idx])])])
    fig.update_xaxes(title_text=f"{letters[i]} &#8594; {letters[j]}", range=[0, 1],
                     tickvals=[0, 1], ticktext=[f"{letters[i]} best", f"{letters[j]} best"],
                     showgrid=False, linecolor="black", row=1, col=2)
    fig.update_yaxes(title_text="P(action)", range=[-0.05, 1], tickvals=[0, 0.5, 1],
                     showgrid=False, linecolor="black",
                     domain=[0.30, 0.76], row=1, col=2)            # shorter, centered vertically
    if len(fig.layout.annotations) > 1:                            # move "P(action) per arm" title down
        fig.layout.annotations[1].update(y=0.79)

    args.outdir.mkdir(parents=True, exist_ok=True)
    lkey = f"L{st['layer']}"
    tag = args.steering.stem.split(lkey, 1)[1] if lkey in args.steering.stem else ""
    out = args.outdir / f"steer_animation_{letters[i]}{letters[j]}_L{args.layer}{tag}.html"
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False,
                   config={"responsive": False, "toImageButtonOptions": {"format": "png", "scale": 6}})
    print(f"K={K} frames; P({letters[i]})/P({letters[j]}): "
          f"{man[0,i]:.2f}/{man[0,j]:.2f} → {man[-1,i]:.2f}/{man[-1,j]:.2f}")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
