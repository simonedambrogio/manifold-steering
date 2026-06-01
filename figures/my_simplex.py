"""Global M_y plot: the behavior (choice-distribution) simplex, the analog of mh_simplex.

Four arms → four "arm-X-is-best" choice-distribution vertices (mean 4-arm choice dist where
X = argmax Q), joined by the six value-transition edges (per-pair, binned by r = Q[j]-Q[i],
fit in Hellinger space exactly as `analyses.fit_pair_manifold`). Drawn the SAME way as
mh_simplex so the two tetrahedra are directly comparable.

4-arm distributions live in the 3-simplex Δ³, so this embeds in 3-D EXACTLY (corners = pure
arms) — no flattening, unlike M_h. Distances/isometry use Hellinger; the picture uses
barycentric coordinates of the raw probabilities (so a vertex = all mass on one arm, and an
edge bowing toward the centre = probability leaking onto the other arms during the handoff).

Two styles via --style:
  default  the original barycentric tetrahedron (gridless scene, bootstrap bands, buttons).
  floor    the color-pair manifold aesthetic (cf. figures/value_relative_structure.py): white
           scene, grey floor, the M_y edges + vertices drawn and shadowed on the floor.

    .venv/bin/python -m figures.my_simplex --style floor --outdir bandit/figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch

from analyses.fit_behavior_manifold import exp_map, log_map
from analyses.fit_pair_manifold import _bin_centroids, _fit_splines, _quantile_edges
from figures.value_relative_structure import (
    ARM_COLORS, FLOOR_COLOR, EDGE_COLOR, SHADOW_COLOR, SHADOW_OPACITY)

# regular-tetrahedron corners for barycentric embedding of a 4-arm distribution
TETRA = np.array([[0, 0, 0], [1, 0, 0],
                  [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])


def _bary(P):
    return np.asarray(P) @ TETRA


def _my_edge(r, C4, n_bins, n_grid):
    """Behavior manifold for one edge: r-binned mean dist → Hellinger spline → dist_grid."""
    e = _quantile_edges(r, n_bins)
    cen, rc, cnt = _bin_centroids(r, C4, e)
    sqrt_c = np.sqrt(cen)
    b = sqrt_c.mean(0); b = b / np.linalg.norm(b)
    tan = log_map(b, sqrt_c)
    _, tg = _fit_splines(rc, tan, np.sqrt(cnt.astype(float)), float(len(rc)), n_grid)
    sphere = exp_map(b, tg)
    grid = sphere ** 2
    grid = grid / grid.sum(1, keepdims=True)
    return grid, cen, rc, cnt


def _hell_arc_chord(grid):
    sg = np.sqrt(grid)
    arc = (np.linalg.norm(np.diff(sg, axis=0), axis=1) / np.sqrt(2)).sum()
    chord = np.linalg.norm(sg[-1] - sg[0]) / np.sqrt(2)
    return arc / max(chord, 1e-9)


def compute(args) -> dict:
    """Behavior vertices, per-edge M_y splines (+ bootstrap grids), and per-arm cloud samples."""
    qd = torch.load(args.q, weights_only=False)
    Q = qd["Q"].cpu().numpy(); n_eps, T, n_arms = Q.shape
    logs = json.loads(args.dataset.read_text())
    letters = logs[0]["letters"]
    choice = np.array([l["choice_dists"] for l in logs], dtype=np.float64)[..., :n_arms]
    choice = choice / choice.sum(-1, keepdims=True)

    Qp = Q[:, 1:].reshape(-1, n_arms)
    Cp = choice[:, 1:].reshape(-1, n_arms)
    order = np.argsort(-Qp, axis=1); argmax = Qp.argmax(1)
    ep_idx = np.repeat(np.arange(n_eps), T - 1)
    vertices = np.stack([Cp[argmax == x].mean(0) for x in range(n_arms)])  # [4, 4]

    rng_cloud = np.random.default_rng(0)
    cloud_samples = [rng_cloud.choice(np.where(argmax == x)[0],
                                      size=min(args.cloud_per_arm, int((argmax == x).sum())),
                                      replace=False) for x in range(n_arms)]

    rng_boot = np.random.default_rng(1)
    edges = {}
    for i, j in itertools.combinations(range(n_arms), 2):
        sel = (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)
        if sel.sum() < args.n_bins * 3:
            continue
        r = (Qp[sel, j] - Qp[sel, i]).astype(np.float64); C4 = Cp[sel]; eps = ep_idx[sel]
        grid, cen, rc, cnt = _my_edge(r, C4, args.n_bins, args.n_grid)
        ue = np.unique(eps); boot = []
        for _ in range(args.n_boot):
            samp = rng_boot.choice(ue, size=len(ue), replace=True)
            sidx = np.concatenate([np.where(eps == ep)[0] for ep in samp])
            boot.append(_my_edge(r[sidx], C4[sidx], args.n_bins, args.n_grid)[0])
        edges[(i, j)] = dict(grid=grid, boot=boot, arc_chord=_hell_arc_chord(grid), n=int(sel.sum()))

    print("M_y per-edge Hellinger arc/chord:",
          {f"{letters[i]}{letters[j]}": round(e["arc_chord"], 2) for (i, j), e in edges.items()})
    print("vertex purity P(best arm):",
          {letters[x]: round(float(vertices[x, x]), 2) for x in range(n_arms)})
    return dict(Cp=Cp, argmax=argmax, vertices=vertices, cloud_samples=cloud_samples,
                edges=edges, letters=letters, n_arms=n_arms)


def build_default(d: dict) -> go.Figure:
    n_arms, letters = d["n_arms"], d["letters"]
    fig = go.Figure()
    cloud_idx = []
    for x in range(n_arms):
        C = _bary(d["Cp"][d["cloud_samples"][x]])
        fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers",
            name=f"{letters[x]} best", marker=dict(size=2, color=ARM_COLORS[x], opacity=0.06)))
        cloud_idx.append(len(fig.data) - 1)

    for k, ((i, j), e) in enumerate(d["edges"].items()):
        for gb in e["boot"]:
            gbb = _bary(gb)
            fig.add_trace(go.Scatter3d(x=gbb[:, 0], y=gbb[:, 1], z=gbb[:, 2], mode="lines",
                line=dict(color="rgba(0,0,0,0.06)", width=1), showlegend=False, hoverinfo="skip"))
        G = _bary(e["grid"])
        fig.add_trace(go.Scatter3d(x=G[:, 0], y=G[:, 1], z=G[:, 2], mode="lines",
            legendgroup="edges", showlegend=(k == 0), name="M_y edge", line=dict(color="black", width=5),
            hovertext=f"{letters[i]}↔{letters[j]} n={e['n']} Hellinger arc/chord={e['arc_chord']:.2f}"))
        ch = _bary(np.stack([e["grid"][0], e["grid"][-1]]))
        fig.add_trace(go.Scatter3d(x=ch[:, 0], y=ch[:, 1], z=ch[:, 2], mode="lines",
            legendgroup="chords", showlegend=(k == 0), name="linear chord",
            line=dict(color="gray", width=2, dash="dash")))

    Vv = _bary(d["vertices"])
    for x in range(n_arms):
        fig.add_trace(go.Scatter3d(x=[Vv[x, 0]], y=[Vv[x, 1]], z=[Vv[x, 2]], mode="markers+text",
            text=[letters[x]], textposition="top center", showlegend=False,
            marker=dict(size=10, color=ARM_COLORS[x], line=dict(color="black", width=2))))

    fig.update_layout(
        title="M_y behavior simplex — choice-distribution tetrahedron (barycentric, exact 3-D); "
              "black=M_y edge (Hellinger spline), dashed=linear mixture, corners=pure-arm choice",
        scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.05, showactive=True, buttons=[
            dict(label="clouds faint", method="restyle", args=[{"marker.opacity": 0.06}, cloud_idx]),
            dict(label="clouds off", method="restyle", args=[{"marker.opacity": 0.0}, cloud_idx])])],
        margin=dict(l=0, r=0, t=70, b=0), legend=dict(x=0, y=1))
    return fig


def build_floor(d: dict, eye=(1.5, 1.5, 1.1), center=(0.0, 0.0, -0.10),
                width=820, height=720, show_title=True, show_buttons=True) -> go.Figure:
    """Color-pair manifold aesthetic: grey floor, M_y edges + vertices skeleton + shadow."""
    n_arms, letters = d["n_arms"], d["letters"]
    Vv = _bary(d["vertices"])
    edge_grids = {k: _bary(e["grid"]) for k, e in d["edges"].items()}
    chords = {k: _bary(np.stack([e["grid"][0], e["grid"][-1]])) for k, e in d["edges"].items()}
    clouds = [_bary(d["Cp"][idx]) for idx in d["cloud_samples"]]

    allpts = np.vstack([Vv] + list(edge_grids.values()))
    ctr = allpts.mean(0)
    scale = max(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1])) + 1e-9
    norm = lambda P: (P - ctr) / scale
    Vv = norm(Vv); edge_grids = {k: norm(g) for k, g in edge_grids.items()}
    chords = {k: norm(c) for k, c in chords.items()}; clouds = [norm(c) for c in clouds]
    aspect = np.ptp(allpts / scale, axis=0)
    z_floor = norm(allpts)[:, 2].min() - 0.12 * (np.ptp(allpts[:, 2] / scale) + 1e-9)

    fig = go.Figure()
    # floor shadows: M_y edges + vertices
    for g in edge_grids.values():
        fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=[z_floor] * len(g), mode="lines",
            line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=Vv[:, 0], y=Vv[:, 1], z=[z_floor] * n_arms, mode="markers",
        marker=dict(size=9, color=SHADOW_COLOR), opacity=SHADOW_OPACITY,
        hoverinfo="skip", showlegend=False))
    # linear chords (dashed) then the M_y edges (black) — the skeleton
    for k, c in enumerate(chords.values()):
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="lines",
            line=dict(color="gray", width=2, dash="dash"), name="linear chord",
            legendgroup="chord", showlegend=(k == 0)))
    for k, g in enumerate(edge_grids.values()):
        fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="lines",
            line=dict(color="black", width=4), name="M_y edge",
            legendgroup="edge", showlegend=(k == 0)))
    n_pre = len(fig.data)                                          # static traces (always shown)

    # faint per-arm choice cloud
    cloud_idx = []
    for x in range(n_arms):
        C = clouds[x]
        fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", showlegend=False,
            name=f"{letters[x]} best", marker=dict(size=2, color=ARM_COLORS[x], opacity=0.08)))
        cloud_idx.append(len(fig.data) - 1)
    # vertices (diamonds)
    for x in range(n_arms):
        fig.add_trace(go.Scatter3d(x=[Vv[x, 0]], y=[Vv[x, 1]], z=[Vv[x, 2]], mode="markers+text",
            text=[letters[x]], textposition="top center", showlegend=False,
            marker=dict(size=12, color=ARM_COLORS[x], line=dict(color="black", width=2))))

    axis = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                showticklabels=False, title="")
    title = dict(
        text="Behaviour simplex M<sub>y</sub>"
             "<br><sup>choice-distribution tetrahedron · black = M<sub>y</sub> edge, "
             "dashed = linear mixture</sup>",
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
            dict(label="clouds faint", method="restyle", args=[{"marker.opacity": 0.08}, cloud_idx]),
            dict(label="clouds off", method="restyle", args=[{"marker.opacity": 0.0}, cloud_idx])])]
            if show_buttons else []),
        legend=dict(x=0, y=0.97 if not show_title else 0.93),
        margin=dict(l=0, r=80, b=0, t=t_margin), width=width, height=height)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q", type=pathlib.Path, default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--n-bins", type=int, default=25)
    p.add_argument("--n-grid", type=int, default=150)
    p.add_argument("--n-boot", type=int, default=15)
    p.add_argument("--cloud-per-arm", type=int, default=1200)
    p.add_argument("--style", choices=["default", "floor"], default="default")
    p.add_argument("--eye", type=float, nargs=3, default=[1.5, 1.5, 1.1],
                   help="floor-style camera eye (x y z); larger magnitude = more zoomed out")
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    d = compute(args)
    if args.style == "floor":
        fig = build_floor(d, eye=tuple(args.eye))
        name = "my_simplex_floor.html"
    else:
        fig = build_default(d)
        name = "my_simplex.html"
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / name
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
