"""Raw unsupervised top-3 PCA of all decision-token activations — no value-informed projection.

The honest baseline: project every trial onto the top-3 principal components of the raw
activation cloud (PCA sees activations only — no Q, no grouping, no decoding). Color by
which-arm-best / confidence / trial-number to ask: does the value structure show up in the
*dominant* variance directions, or is it buried? (Expectation: buried — value is ~5% of the
variance, so the top PCs are dominated by nuisance like trial-phase.)

Prints, per top PC, its correlation with trial-number (nuisance proxy) vs Q_max / margin.

Two styles via --style:
  default  the original clean-cloud scene (gridded PCA axes, toggle buttons).
  floor    the color-pair manifold aesthetic (cf. figures/value_relative_structure.py): white
           scene, grey floor, the M_h value-manifold edges + which-best vertices drawn and
           shadowed on the floor, data-aspect box, angled camera.

    .venv/bin/python -m figures.raw_pca3 --layer 19 --trial-min 95 --trial-max 106 --style floor
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch

from analyses.fit_pair_manifold import _bin_centroids, _fit_splines, _quantile_edges
from figures.value_relative_structure import (
    ARM_COLORS, FLOOR_COLOR, EDGE_COLOR, SHADOW_COLOR, SHADOW_OPACITY)


def compute(args) -> dict:
    """Raw top-3 PCA of all trials + which-best labels, vertices, and M_h edge splines."""
    act = torch.load(args.activations, weights_only=False)
    ep_ids = act["episode_ids"]; A_all = act["activations"]
    n_eps, T, _, hid = A_all.shape
    tmin = max(1, args.trial_min); tmax = args.trial_max if args.trial_max > 0 else T
    A = A_all[:, tmin:tmax, args.layer, :].float().reshape(-1, hid); del A_all, act
    Q = torch.load(args.q, weights_only=False)["Q"][ep_ids].cpu().numpy()
    na = Q.shape[2]
    letters = json.loads(args.dataset.read_text())[0]["letters"]

    mu = A.mean(0)
    _, S, Vh = torch.linalg.svd(A - mu, full_matrices=False)
    coords = ((A - mu) @ Vh[:3].T).cpu().numpy()
    var3 = ((S[:3] ** 2) / (S ** 2).sum()).cpu().numpy()

    Qp = Q[:, tmin:tmax].reshape(-1, na)
    argmax = Qp.argmax(1)
    srt = np.sort(Qp, axis=1)[:, ::-1]
    conf = srt[:, 0] - srt[:, 1]
    Qmax = srt[:, 0]
    trial = np.tile(np.arange(tmin, tmax), n_eps).astype(float)
    rng_txt = "all trials" if tmax - tmin > T - 5 else f"trials {tmin}–{tmax-1} (N={len(A)})"
    print(f"[{rng_txt}]")

    def cc(a, b):
        return np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    print(f"raw top-3 PCA var explained: {var3.round(3).tolist()} (sum {var3.sum():.2f})")
    print("what do the top PCs encode? Pearson r:")
    for k in range(3):
        c = coords[:, k]
        print(f"  PC{k+1}: trial# {cc(c,trial):+.2f}   Q_max {cc(c,Qmax):+.2f}   margin {cc(c,conf):+.2f}")

    # overlay: which-best vertex centroids + value-manifold edges, in the raw top-3 coords
    order = np.argsort(-Qp, axis=1)
    vert = np.stack([coords[argmax == x].mean(0) for x in range(na)])
    edges = {}
    for i, j in itertools.combinations(range(na), 2):
        sel = (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)
        if sel.sum() < args.n_bins * 3:
            continue
        r = (Qp[sel, j] - Qp[sel, i]).astype(float)
        cen, rc, cnt = _bin_centroids(r, coords[sel], _quantile_edges(r, args.n_bins))
        _, g = _fit_splines(rc, cen, np.sqrt(cnt.astype(float)), float(len(rc)), 120)
        edges[(i, j)] = g
    pred = ((coords[:, None, :] - vert[None]) ** 2).sum(2).argmin(1)
    acc = float((pred == argmax).mean())
    print(f"which-best nearest-centroid accuracy in raw top-3: {acc:.2f} (chance {1/na:.2f})")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(coords), size=min(args.n_show, len(coords)), replace=False)
    return dict(coords=coords, idx=idx, C=coords[idx], argmax_idx=argmax[idx], conf_idx=conf[idx],
                trial_idx=trial[idx], vertices=vert, edges=edges, letters=letters, na=na,
                var3=var3, layer=args.layer, rng_txt=rng_txt, acc=acc)


def build_default(d: dict) -> go.Figure:
    C, na, letters, var3 = d["C"], d["na"], d["letters"], d["var3"]
    fig = go.Figure()
    cat_idx = []
    for x in range(na):
        m = d["argmax_idx"] == x
        fig.add_trace(go.Scatter3d(x=C[m, 0], y=C[m, 1], z=C[m, 2], mode="markers",
            name=f"{letters[x]} best", marker=dict(size=2.5, color=ARM_COLORS[x], opacity=0.45)))
        cat_idx.append(len(fig.data) - 1)
    conf_i = len(fig.data)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", visible=False,
        name="confidence", marker=dict(size=2.5, color=d["conf_idx"], colorscale="Plasma", opacity=0.5,
        colorbar=dict(title="margin"))))
    tr_i = len(fig.data)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", visible=False,
        name="trial #", marker=dict(size=2.5, color=d["trial_idx"], colorscale="Cividis", opacity=0.5,
        colorbar=dict(title="trial #"))))
    overlay_idx = []
    for k, ((i, j), g) in enumerate(d["edges"].items()):
        fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="lines",
            line=dict(color="black", width=4), name="M_h edge", legendgroup="edge",
            showlegend=(k == 0)))
        overlay_idx.append(len(fig.data) - 1)
    for x in range(na):
        v = d["vertices"][x]
        fig.add_trace(go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode="markers+text",
            text=[letters[x]], textposition="top center", showlegend=False,
            marker=dict(size=8, color=ARM_COLORS[x], line=dict(color="black", width=2))))
        overlay_idx.append(len(fig.data) - 1)
    n_tr = len(fig.data)

    def vis(mode):
        v = [False] * n_tr
        for k in overlay_idx:
            v[k] = True
        if mode == "cat":
            for k in cat_idx:
                v[k] = True
        elif mode == "conf":
            v[conf_i] = True
        else:
            v[tr_i] = True
        return v

    fig.update_layout(
        title=(f"Raw unsupervised top-3 PCA — {d['rng_txt']} (L{d['layer']}) — var {var3.sum():.0%}. "
               "PCA sees activations only (no Q)."
               "<br><sup>Fixing the trial removes trial-phase variance; does which-best separate now? "
               "If not, the value code is genuinely low-variance.</sup>"),
        scene=dict(xaxis_title=f"PC1 ({var3[0]:.0%})", yaxis_title=f"PC2 ({var3[1]:.0%})",
                   zaxis_title=f"PC3 ({var3[2]:.0%})"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.06, showactive=True, buttons=[
            dict(label="which-best", method="update", args=[{"visible": vis("cat")}]),
            dict(label="confidence", method="update", args=[{"visible": vis("conf")}]),
            dict(label="trial #", method="update", args=[{"visible": vis("tr")}])])],
        margin=dict(l=0, r=0, t=70, b=0), legend=dict(x=0, y=1))
    return fig


def build_floor(d: dict, eye=(1.5, 1.5, 1.1), center=(0.0, 0.0, -0.10),
                width=820, height=720, show_title=True, show_buttons=True) -> go.Figure:
    """Color-pair manifold aesthetic: grey floor, M_h edges + vertices skeleton + shadow."""
    na, letters, var3 = d["na"], d["letters"], d["var3"]
    # normalize cloud + vertices + edge splines together (matched-zoom scale, honest aspect)
    C = d["C"].astype(float).copy(); V = d["vertices"].astype(float).copy()
    splines = {k: g.astype(float).copy() for k, g in d["edges"].items()}
    ctr = C.mean(0)
    scale = max(np.ptp(C[:, 0] - ctr[0]), np.ptp(C[:, 1] - ctr[1])) + 1e-9
    norm = lambda P: (P - ctr) / scale
    C = norm(C); V = norm(V); splines = {k: norm(g) for k, g in splines.items()}
    aspect = np.ptp(C, axis=0)
    z_floor = C[:, 2].min() - 0.12 * (np.ptp(C[:, 2]) + 1e-9)

    fig = go.Figure()
    # floor shadows: M_h edges + vertices projected down
    for g in splines.values():
        fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=[z_floor] * len(g), mode="lines",
            line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=V[:, 0], y=V[:, 1], z=[z_floor] * na, mode="markers",
        marker=dict(size=9, color=SHADOW_COLOR), opacity=SHADOW_OPACITY,
        hoverinfo="skip", showlegend=False))
    # the M_h value-manifold edges (the skeleton) in 3D
    for k, g in enumerate(splines.values()):
        fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="lines",
            line=dict(color="#222222", width=3), name="M_h edge", legendgroup="edge",
            showlegend=(k == 0)))
    n_pre = len(fig.data)                                           # static traces (always shown)

    # the belief cloud (toggled colorings)
    cat_idx = []
    for x in range(na):
        m = d["argmax_idx"] == x
        fig.add_trace(go.Scatter3d(x=C[m, 0], y=C[m, 1], z=C[m, 2], mode="markers", showlegend=False,
            name=f"{letters[x]} best", marker=dict(size=2.5, color=ARM_COLORS[x], opacity=0.22)))
        cat_idx.append(len(fig.data) - 1)
    conf_i = len(fig.data)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", visible=False,
        name="confidence", showlegend=False, marker=dict(size=2.5, color=d["conf_idx"],
        colorscale="Plasma", opacity=0.3, colorbar=dict(title="margin", x=1.02, thickness=18))))
    tr_i = len(fig.data)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", visible=False,
        name="trial #", showlegend=False, marker=dict(size=2.5, color=d["trial_idx"],
        colorscale="Cividis", opacity=0.3, colorbar=dict(title="trial #", x=1.02, thickness=18))))
    # which-best vertices (diamonds)
    for x in range(na):
        fig.add_trace(go.Scatter3d(x=[V[x, 0]], y=[V[x, 1]], z=[V[x, 2]], mode="markers+text",
            text=[letters[x]], textposition="top center", showlegend=False,
            marker=dict(size=12, color=ARM_COLORS[x], line=dict(color="black", width=2))))
    n_tr, n_vert = len(fig.data), na

    def vis(mode):
        v = [False] * n_tr
        for k in range(n_pre):
            v[k] = True
        for k in range(n_vert):
            v[n_tr - n_vert + k] = True
        if mode == "cat":
            for k in cat_idx:
                v[k] = True
        elif mode == "conf":
            v[conf_i] = True
        else:
            v[tr_i] = True
        return v

    axis = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                showticklabels=False, title="")
    title = dict(
        text=f"Raw top-3 PCA — {d['rng_txt']} (L{d['layer']})"
             f"<br><sup>which-best vertices + M_h edges in unsupervised geometry · "
             f"nearest-centroid {d['acc']:.0%}</sup>",
        x=0.5, xanchor="center", font=dict(size=22)) if show_title else None
    t_margin = 70 if show_title else 34
    fig.update_layout(
        title=title, font=dict(size=15),
        scene=dict(xaxis=axis, yaxis=axis, zaxis={**axis, "backgroundcolor": FLOOR_COLOR},
                   xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=True,
                   aspectmode="manual",
                   aspectratio=dict(x=float(aspect[0]), y=float(aspect[1]), z=float(aspect[2])),
                   camera=dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
                               center=dict(x=float(center[0]), y=float(center[1]), z=float(center[2])))),
        paper_bgcolor="white",
        updatemenus=([dict(type="buttons", direction="right", x=0, y=1.0, showactive=True, buttons=[
            dict(label="which-best", method="update", args=[{"visible": vis("cat")}]),
            dict(label="confidence", method="update", args=[{"visible": vis("conf")}]),
            dict(label="trial #", method="update", args=[{"visible": vis("tr")}])])]
            if show_buttons else []),
        legend=dict(x=0, y=0.97 if not show_title else 0.93),
        margin=dict(l=0, r=80, b=0, t=t_margin), width=width, height=height)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", type=pathlib.Path,
                   default=pathlib.Path("artifacts/activations/llama8b_fixed_residual.pt"))
    p.add_argument("--q", type=pathlib.Path, default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--trial-min", type=int, default=1, help="restrict to trials [min,max) — fixing")
    p.add_argument("--trial-max", type=int, default=0, help="trial removes trial-phase variance; 0=all")
    p.add_argument("--n-show", type=int, default=6000)
    p.add_argument("--n-bins", type=int, default=12, help="r-bins per edge for the manifold overlay")
    p.add_argument("--style", choices=["default", "floor"], default="default")
    p.add_argument("--eye", type=float, nargs=3, default=[1.5, 1.5, 1.1],
                   help="floor-style camera eye (x y z); larger magnitude = more zoomed out")
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    d = compute(args)
    T = 150
    suffix = "" if (args.trial_max if args.trial_max > 0 else T) - max(1, args.trial_min) > T - 5 \
        else f"_t{max(1, args.trial_min)}-{(args.trial_max if args.trial_max > 0 else T) - 1}"
    if args.style == "floor":
        fig = build_floor(d, eye=tuple(args.eye))
        name = f"raw_pca3_floor_L{args.layer}{suffix}.html"
    else:
        fig = build_default(d)
        name = f"raw_pca3_L{args.layer}{suffix}.html"
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / name
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
