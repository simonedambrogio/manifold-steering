"""Activation manifold in CHOICE-RELEVANT relative-value coordinates (exploratory).

The choice depends on Q only through the centered vector q_c = Q − mean(Q) (sum-zero, 3-D).
We decode Q from activations, take the 3-D subspace spanned by the centered decoding weights
(the directions of activation space that carry choice-relevant value), and project ALL trials
into it (not just the top-2 pairs). This shows the full belief geometry — including the interior
(3+ arms comparable) and any confidence/radial structure that the top-2 edge simplex drops.

Color toggle: which-arm-best (4 categories) | confidence (Q_max−Q_2nd) | entropy of softmax(βQ).
Structure probes printed: variance captured by the value subspace; corr(radial distance,
confidence) — i.e. is "distance from the uncertain center" the confidence axis?

Two styles via --style:
  default  the original clean-cloud scene (gridded PCA axes, toggle buttons).
  floor    the color-pair manifold aesthetic (colorpair/figures/fig_pca3d): white scene,
           grey floor plane, the 4-arm simplex skeleton drawn + shadowed on the floor,
           data-aspect box, angled camera.

    .venv/bin/python -m figures.value_relative_structure --layer 19 --style floor --outdir bandit/figures
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch

ARM_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

# color-pair manifold palette (matches colorpair/figures/fig_pca3d.py)
FLOOR_COLOR = "#e5e5e5"
EDGE_COLOR = "rgba(100,100,100,0.45)"
SHADOW_COLOR = "#666666"
SHADOW_OPACITY = 0.15


def compute(args) -> dict:
    """Project all trials into the 3-D choice-relevant value subspace + per-trial labels."""
    act = torch.load(args.activations, weights_only=False)
    ep_ids = act["episode_ids"]; A_all = act["activations"]
    n_eps, T, _, hid = A_all.shape
    A = A_all[:, 1:, args.layer, :].float().reshape(-1, hid); del A_all, act
    Q = torch.load(args.q, weights_only=False)["Q"][ep_ids].cpu().numpy()
    na = Q.shape[2]
    letters = json.loads(args.dataset.read_text())[0]["letters"]

    mu = A.mean(0); _, _, Vh = torch.linalg.svd(A - mu, full_matrices=False)
    X = ((A - mu) @ Vh[: args.pca_dim].T).cpu().numpy()
    Xs = (X - X.mean(0)) / X.std(0)
    Qp = Q[:, 1:].reshape(-1, na)

    # decode all 4 arm values; centered weights span the choice-relevant value subspace
    lam = 50.0
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(args.pca_dim), Xs.T @ (Qp - Qp.mean(0)))  # [64,4]
    Wc = (W - W.mean(1, keepdims=True)).T                                                  # [4,64] centered
    _, S, Vv = np.linalg.svd(Wc, full_matrices=False)
    B = Vv[:3]                                                                              # [3,64] value subspace
    coords = Xs @ B.T                                                                       # [N,3]
    var_in_sub = coords.var(0).sum() / Xs.var(0).sum()

    argmax = Qp.argmax(1)
    srt = np.sort(Qp, axis=1)[:, ::-1]
    conf = srt[:, 0] - srt[:, 1]                                                            # margin = confidence
    bq = args.beta * Qp; bq -= bq.max(1, keepdims=True)
    sm = np.exp(bq); sm /= sm.sum(1, keepdims=True)
    ent = -(sm * np.log(sm + 1e-12)).sum(1)                                                 # entropy
    radial = np.linalg.norm(coords - coords.mean(0), axis=1)
    r_conf = np.corrcoef(radial, conf)[0, 1]
    r_ent = np.corrcoef(radial, ent)[0, 1]
    print(f"value subspace captures {var_in_sub:.1%} of activation variance")
    print(f"corr(radial distance, confidence/margin) = {r_conf:+.2f}   "
          f"corr(radial, entropy) = {r_ent:+.2f}")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(coords), size=min(args.n_show, len(coords)), replace=False)
    vertices = np.stack([coords[argmax == x].mean(0) for x in range(na)])                   # which-best centroids
    return dict(coords=coords, idx=idx, C=coords[idx], argmax_idx=argmax[idx],
                conf_idx=conf[idx], ent_idx=ent[idx], vertices=vertices, letters=letters,
                na=na, var_in_sub=var_in_sub, r_conf=r_conf, layer=args.layer)


def build_default(d: dict) -> go.Figure:
    C, na, letters = d["C"], d["na"], d["letters"]
    fig = go.Figure()
    cat_idx = []
    for x in range(na):
        m = d["argmax_idx"] == x
        fig.add_trace(go.Scatter3d(x=C[m, 0], y=C[m, 1], z=C[m, 2], mode="markers",
            name=f"{letters[x]} best", marker=dict(size=2.5, color=ARM_COLORS[x], opacity=0.45)))
        cat_idx.append(len(fig.data) - 1)
    conf_i = len(fig.data)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", visible=False,
        name="confidence", marker=dict(size=2.5, color=d["conf_idx"], colorscale="Plasma",
        opacity=0.5, colorbar=dict(title="margin"))))
    ent_i = len(fig.data)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", visible=False,
        name="entropy", marker=dict(size=2.5, color=d["ent_idx"], colorscale="Viridis",
        opacity=0.5, colorbar=dict(title="entropy"))))
    for x in range(na):
        v = d["vertices"][x]
        fig.add_trace(go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode="markers+text",
            text=[letters[x]], textposition="top center", showlegend=False,
            marker=dict(size=9, color=ARM_COLORS[x], line=dict(color="black", width=2))))
    n_tr, n_vert = len(fig.data), na

    def vis(mode):
        v = [False] * n_tr
        for k in range(n_vert):
            v[n_tr - n_vert + k] = True
        if mode == "cat":
            for k in cat_idx:
                v[k] = True
        elif mode == "conf":
            v[conf_i] = True
        else:
            v[ent_i] = True
        return v

    fig.update_layout(
        title=(f"Choice-relevant relative-value structure (L{d['layer']}) — all trials in the "
               f"value subspace ({d['var_in_sub']:.0%} of var). corr(radial, confidence)={d['r_conf']:+.2f}"
               "<br><sup>axes = orthogonal directions of the choice-relevant value subspace; "
               "position = relative-value belief, NOT a single arm's value</sup>"),
        scene=dict(xaxis_title="rel-value PC1", yaxis_title="rel-value PC2",
                   zaxis_title="rel-value PC3"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.06, showactive=True, buttons=[
            dict(label="which-best", method="update", args=[{"visible": vis("cat")}]),
            dict(label="confidence", method="update", args=[{"visible": vis("conf")}]),
            dict(label="entropy", method="update", args=[{"visible": vis("ent")}])])],
        margin=dict(l=0, r=0, t=70, b=0), legend=dict(x=0, y=1))
    return fig


def _segments(points: np.ndarray, pairs, z=None):
    """Flatten point-pairs into None-separated x/y/z line segments (optionally flat at z)."""
    xs, ys, zs = [], [], []
    for i, j in pairs:
        xs += [points[i, 0], points[j, 0], None]
        ys += [points[i, 1], points[j, 1], None]
        zs += [z if z is not None else points[i, 2], z if z is not None else points[j, 2], None]
    return xs, ys, zs


def build_floor(d: dict, eye=(1.5, 1.5, 1.1), center=(-0.22, 0.0, -0.14),
                width=820, height=720, show_title=True, show_buttons=True) -> go.Figure:
    """Color-pair manifold aesthetic: grey floor, simplex skeleton + shadow, data-aspect box.

    `center` shifts the camera look-at to recentre the cloud (the floor extends the scene box
    downward, which otherwise pushes the cloud up-and-right).
    """
    na, letters = d["na"], d["letters"]
    # normalize cloud + vertices together (matched-zoom scale, honest aspect) as in fig_pca3d
    C = d["C"].astype(float).copy(); V = d["vertices"].astype(float).copy()
    ctr = C.mean(0); C -= ctr; V -= ctr
    scale = max(np.ptp(C[:, 0]), np.ptp(C[:, 1])) + 1e-9
    C /= scale; V /= scale
    aspect = np.ptp(C, axis=0)
    z_floor = C[:, 2].min() - 0.12 * (np.ptp(C[:, 2]) + 1e-9)
    pairs = [(i, j) for i in range(na) for j in range(i + 1, na)]   # 6 simplex edges

    fig = go.Figure()
    # floor shadows: simplex edges + vertices projected down
    sx, sy, sz = _segments(V, pairs, z=z_floor)
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines",
        line=dict(color=SHADOW_COLOR, width=2), opacity=SHADOW_OPACITY,
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=V[:, 0], y=V[:, 1], z=[z_floor] * na, mode="markers",
        marker=dict(size=9, color=SHADOW_COLOR), opacity=SHADOW_OPACITY,
        hoverinfo="skip", showlegend=False))
    # the simplex skeleton in 3D
    ex, ey, ez = _segments(V, pairs)
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
        line=dict(color=EDGE_COLOR, width=4), hoverinfo="skip", showlegend=False))
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
    ent_i = len(fig.data)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", visible=False,
        name="entropy", showlegend=False, marker=dict(size=2.5, color=d["ent_idx"],
        colorscale="Viridis", opacity=0.3, colorbar=dict(title="entropy", x=1.02, thickness=18))))
    # the simplex vertices (diamonds, like the color-pair colored markers)
    for x in range(na):
        fig.add_trace(go.Scatter3d(x=[V[x, 0]], y=[V[x, 1]], z=[V[x, 2]], mode="markers+text",
            text=[letters[x]], textposition="top center", showlegend=False,
            marker=dict(size=12, color=ARM_COLORS[x], line=dict(color="black", width=2))))
    n_tr, n_vert = len(fig.data), na

    def vis(mode):
        v = [False] * n_tr
        for k in range(n_pre):
            v[k] = True                                            # shadows + skeleton
        for k in range(n_vert):
            v[n_tr - n_vert + k] = True                            # vertices
        if mode == "cat":
            for k in cat_idx:
                v[k] = True
        elif mode == "conf":
            v[conf_i] = True
        else:
            v[ent_i] = True
        return v

    axis = dict(showgrid=False, zeroline=False, showline=False, backgroundcolor="white",
                showticklabels=False, title="")
    title = dict(
        text=f"Relative-value belief manifold (L{d['layer']})"
             f"<br><sup>4-arm simplex · {d['var_in_sub']:.0%} of activation variance</sup>",
        x=0.5, xanchor="center", font=dict(size=24)) if show_title else None
    t_margin = 70 if show_title else 34            # no title → reclaim the top for the scene
    fig.update_layout(
        title=title,
        font=dict(size=15),
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
            dict(label="entropy", method="update", args=[{"visible": vis("ent")}])])]
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
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.016, help="softmax temp for entropy (BestRL β)")
    p.add_argument("--n-show", type=int, default=6000)
    p.add_argument("--style", choices=["default", "floor"], default="default")
    p.add_argument("--eye", type=float, nargs=3, default=[1.5, 1.5, 1.1],
                   help="floor-style camera eye (x y z); larger magnitude = more zoomed out")
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    d = compute(args)
    if args.style == "floor":
        fig = build_floor(d, eye=tuple(args.eye))
        name = f"value_relative_structure_floor_L{args.layer}.html"
    else:
        fig = build_default(d)
        name = f"value_relative_structure_L{args.layer}.html"
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / name
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
