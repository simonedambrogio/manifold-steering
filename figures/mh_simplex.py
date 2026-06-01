"""Global M_h plot: the value-concept simplex (all arms + edges) in ONE shared 3-D space.

Four arms → four "arm-X-is-best" vertices (mean decision-token activation where X = argmax Q),
joined by the six value-transition edges (per-pair top-2 splines in r = Q[j]-Q[i]). All edges
share ONE projection (skeleton PCA-3) so the tetrahedron is visible and directly comparable to
the behavior simplex M_y we draw the same way.

FIDELITY NOTE: a single linear 3-D frame captures the 4 vertices fully but flattens each edge's
bend to ~80-87% (the curved simplex is intrinsically a bit >3-D). Per-edge fidelity is printed
and titled; the fully-faithful per-edge arcs are in `figures.mh_simplex_faithful`. Curvature
itself is the 64-D arc/chord, reported per edge.

Clean treatment: crimson edge splines, dashed linear chords, per-edge episode-bootstrap bands,
labeled vertices, faint (toggleable) clouds colored by which-arm-is-best.

    .venv/bin/python -m figures.mh_simplex --layer 19 --outdir bandit/figures
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

ARM_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]  # B, C, D, F


def _spline(r, X, n_bins, n_grid):
    e = _quantile_edges(r, n_bins)
    cen, rc, cnt = _bin_centroids(r, X, e)
    _, grid = _fit_splines(rc, cen, np.sqrt(cnt.astype(float)), float(len(rc)), n_grid)
    return grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", type=pathlib.Path,
                   default=pathlib.Path("artifacts/activations/llama8b_fixed_residual.pt"))
    p.add_argument("--q", type=pathlib.Path, default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=25)
    p.add_argument("--n-grid", type=int, default=150)
    p.add_argument("--n-boot", type=int, default=15)
    p.add_argument("--cloud-per-arm", type=int, default=1200)
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    act = torch.load(args.activations, weights_only=False)
    ep_ids = act["episode_ids"]; A_all = act["activations"]
    n_eps, T, _, hidden = A_all.shape
    A = A_all[:, 1:, args.layer, :].float().reshape(-1, hidden); del A_all, act
    qd = torch.load(args.q, weights_only=False)
    Q = qd["Q"][ep_ids].cpu().numpy(); n_arms = Q.shape[2]
    letters = json.loads(args.dataset.read_text())[0]["letters"]

    A_mean = A.mean(dim=0)
    _, _, Vh = torch.linalg.svd(A - A_mean, full_matrices=False)
    A_pca = ((A - A_mean) @ Vh[: args.pca_dim].T).cpu().numpy()
    Qp = Q[:, 1:].reshape(-1, n_arms); order = np.argsort(-Qp, axis=1)
    argmax = Qp.argmax(1); ep_idx = np.repeat(np.arange(n_eps), T - 1)
    vertices = np.stack([A_pca[argmax == x].mean(0) for x in range(n_arms)])

    edges = {}
    for i, j in itertools.combinations(range(n_arms), 2):
        sel = (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)
        if sel.sum() < args.n_bins * 3:
            continue
        r = (Qp[sel, j] - Qp[sel, i]).astype(np.float64)
        grid = _spline(r, A_pca[sel], args.n_bins, args.n_grid)
        arc = np.linalg.norm(np.diff(grid, axis=0), axis=1).sum()
        edges[(i, j)] = dict(r=r, X=A_pca[sel], eps=ep_idx[sel], grid=grid,
                             arc_chord=arc / np.linalg.norm(grid[-1] - grid[0]), n=int(sel.sum()))

    # shared projection: PCA-3 of the skeleton (vertices + edge grids)
    skel = np.vstack([vertices] + [e["grid"] for e in edges.values()])
    mu3 = skel.mean(0)
    _, S3, V3 = np.linalg.svd(skel - mu3, full_matrices=False)
    W3 = V3[:3]; var3 = (S3[:3] ** 2) / (S3 ** 2).sum()
    to3 = lambda M: (M - mu3) @ W3.T

    fig = go.Figure()
    cloud_idx = []
    rng = np.random.default_rng(0)
    for x in range(n_arms):
        idx = np.where(argmax == x)[0]
        idx = rng.choice(idx, size=min(args.cloud_per_arm, len(idx)), replace=False)
        C = to3(A_pca[idx])
        fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers",
            name=f"{letters[x]} best", marker=dict(size=2, color=ARM_COLORS[x], opacity=0.06)))
        cloud_idx.append(len(fig.data) - 1)

    fid = {}
    for k, ((i, j), e) in enumerate(edges.items()):
        # per-edge fidelity of the shared frame
        g = e["grid"]; gs = (g - mu3) @ W3.T
        fid[(i, j)] = gs.var(0).sum() / (g - g.mean(0)).var(0).sum()
        # bootstrap band (resample episodes), projected in the shared frame
        ue = np.unique(e["eps"])
        for b in range(args.n_boot):
            samp = rng.choice(ue, size=len(ue), replace=True)
            sidx = np.concatenate([np.where(e["eps"] == ep)[0] for ep in samp])
            gb = to3(_spline(e["r"][sidx], e["X"][sidx], args.n_bins, args.n_grid))
            fig.add_trace(go.Scatter3d(x=gb[:, 0], y=gb[:, 1], z=gb[:, 2], mode="lines",
                line=dict(color="rgba(0,0,0,0.06)", width=1), showlegend=False, hoverinfo="skip"))
        G = to3(g)
        fig.add_trace(go.Scatter3d(x=G[:, 0], y=G[:, 1], z=G[:, 2], mode="lines",
            legendgroup="edges", showlegend=(k == 0), name="M_h edge",
            line=dict(color="black", width=5),
            hovertext=f"{letters[i]}↔{letters[j]} n={e['n']} arc/chord={e['arc_chord']:.2f} "
                      f"shared-fidelity={fid[(i,j)]:.0%}"))
        ch = to3(np.stack([vertices[i], vertices[j]]))
        fig.add_trace(go.Scatter3d(x=ch[:, 0], y=ch[:, 1], z=ch[:, 2], mode="lines",
            legendgroup="chords", showlegend=(k == 0), name="linear chord",
            line=dict(color="gray", width=2, dash="dash")))

    Vv = to3(vertices)
    for x in range(n_arms):
        fig.add_trace(go.Scatter3d(x=[Vv[x, 0]], y=[Vv[x, 1]], z=[Vv[x, 2]], mode="markers+text",
            text=[letters[x]], textposition="top center", showlegend=False,
            marker=dict(size=10, color=ARM_COLORS[x], line=dict(color="black", width=2))))

    fmin, fmax = min(fid.values()), max(fid.values())
    fig.update_layout(
        title=f"M_h value simplex (L{args.layer}) — shared PCA-3, skeleton var {var3.sum():.0%}; "
              f"per-edge fidelity {fmin:.0%}–{fmax:.0%} (bends flattened ~{1-fmax:.0%}–{1-fmin:.0%})",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.05, showactive=True, buttons=[
            dict(label="clouds faint", method="restyle", args=[{"marker.opacity": 0.06}, cloud_idx]),
            dict(label="clouds off", method="restyle", args=[{"marker.opacity": 0.0}, cloud_idx])])],
        margin=dict(l=0, r=0, t=70, b=0), legend=dict(x=0, y=1))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"mh_simplex_L{args.layer}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print("per-edge shared-frame fidelity:",
          {f"{letters[i]}{letters[j]}": round(float(v), 2) for (i, j), v in fid.items()})
    print(f"skeleton PCA-3 var {var3.sum():.2f}; saved → {out}")


if __name__ == "__main__":
    main()
