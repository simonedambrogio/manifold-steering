"""Faithful all-pairs M_h: each of the 6 edges in its OWN manifold-aligned 2-D frame.

A single shared 3-D projection of the simplex (mh_simplex) is faithful to the 4 vertices but
flattens each edge's curvature (the edges bend out of the 3-D vertex-space). This figure shows
every edge faithfully instead: a 2x3 grid where each panel is one edge in its own PCA-2 frame
(mPC1 = value progression, mPC2 = bend), with low-opacity cloud, centroids, the M_h spline, an
episode-bootstrap band, and the linear chord. Prints per-edge fidelity: own-2D var vs the var
that survives in the shared 3-D simplex view.

    .venv/bin/python -m figures.mh_simplex_faithful --layer 19 --outdir bandit/figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from analyses.fit_pair_manifold import _bin_centroids, _fit_splines, _quantile_edges


def _spline(r, X, n_bins, n_grid):
    e = _quantile_edges(r, n_bins)
    cen, rc, cnt = _bin_centroids(r, X, e)
    _, grid = _fit_splines(rc, cen, np.sqrt(cnt.astype(float)), float(len(rc)), n_grid)
    return grid, cen, rc, cnt


def _pcak(X, k):
    mu = X.mean(0)
    _, S, V = np.linalg.svd(X - mu, full_matrices=False)
    return mu, V[:k], (S[:k] ** 2) / (S ** 2 + 1e-12).sum()


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
    p.add_argument("--cloud-per-edge", type=int, default=900)
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

    # collect edges (64-D splines) for both the faithful grid and the shared-frame fidelity report
    E = {}
    for i, j in itertools.combinations(range(n_arms), 2):
        sel = (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)
        if sel.sum() < args.n_bins * 3:
            continue
        r = (Qp[sel, j] - Qp[sel, i]).astype(np.float64)
        grid, cen, rc, cnt = _spline(r, A_pca[sel], args.n_bins, args.n_grid)
        E[(i, j)] = dict(r=r, X=A_pca[sel], eps=ep_idx[sel], grid=grid, cen=cen, rc=rc, cnt=cnt)

    # shared-frame fidelity: PCA-3 of the skeleton (vertices + edges)
    skel = np.vstack([vertices] + [e["grid"] for e in E.values()])
    smu, sW, _ = _pcak(skel, 3)
    print(f"{'edge':6} {'n':>5} {'arc/chord_64':>12} {'own-2D var':>11} {'shared-3D var':>13}")
    for (i, j), e in E.items():
        g = e["grid"]
        arc = np.linalg.norm(np.diff(g, axis=0), axis=1).sum()
        chord = np.linalg.norm(g[-1] - g[0])
        _, _, v2 = _pcak(g, 2)
        gs = (g - smu) @ sW.T
        shared_var = gs.var(0).sum() / (g - g.mean(0)).var(0).sum()
        e["arc_chord"] = arc / chord; e["own2_var"] = float(v2.sum())
        print(f"{letters[i]}-{letters[j]:3} {e['X'].shape[0]:5d} {arc/chord:12.2f} "
              f"{v2.sum():11.2f} {shared_var:13.2f}")

    # ---- faithful 2x3 grid: each edge in its own PCA-2 frame ----
    keys = list(E.keys())
    titles = [f"{letters[i]}↔{letters[j]} (arc/chord {E[(i,j)]['arc_chord']:.2f})" for i, j in keys]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.06,
                        vertical_spacing=0.10)
    rng = np.random.default_rng(0)
    for k, (i, j) in enumerate(keys):
        e = E[(i, j)]; row, col = k // 3 + 1, k % 3 + 1
        mu2, W2, _ = _pcak(e["grid"], 2)
        to2 = lambda M: (M - mu2) @ W2.T
        r, X, eps = e["r"], e["X"], e["eps"]
        # cloud (subsampled, low opacity, colored by r) — colorbar only once
        idx = rng.choice(len(r), size=min(args.cloud_per_edge, len(r)), replace=False)
        P = to2(X[idx])
        fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", showlegend=False,
            marker=dict(size=3, color=r[idx], colorscale="Turbo", opacity=0.18,
                        showscale=(k == 0), colorbar=dict(title="r", x=1.02)),
            hoverinfo="skip"), row, col)
        # bootstrap band (resample episodes), projected in this edge's fixed frame
        ue = np.unique(eps)
        for b in range(args.n_boot):
            samp = rng.choice(ue, size=len(ue), replace=True)
            sidx = np.concatenate([np.where(eps == ep)[0] for ep in samp])
            gb, *_ = _spline(r[sidx], X[sidx], args.n_bins, args.n_grid); gb = to2(gb)
            fig.add_trace(go.Scatter(x=gb[:, 0], y=gb[:, 1], mode="lines", showlegend=False,
                line=dict(color="rgba(220,20,60,0.10)", width=1), hoverinfo="skip"), row, col)
        # chord + spline + centroids
        lo = X[r <= np.quantile(r, 0.1)].mean(0); hi = X[r >= np.quantile(r, 0.9)].mean(0)
        ch = to2(np.stack([lo, hi]))
        fig.add_trace(go.Scatter(x=ch[:, 0], y=ch[:, 1], mode="lines", showlegend=False,
            line=dict(color="black", width=2, dash="dash")), row, col)
        G = to2(e["grid"])
        fig.add_trace(go.Scatter(x=G[:, 0], y=G[:, 1], mode="lines", showlegend=False,
            line=dict(color="crimson", width=4)), row, col)
        C = to2(e["cen"])
        fig.add_trace(go.Scatter(x=C[:, 0], y=C[:, 1], mode="markers", showlegend=False,
            marker=dict(size=4 + 8 * np.sqrt(e["cnt"] / e["cnt"].max()), color=e["rc"],
                        colorscale="Turbo", line=dict(color="white", width=1))), row, col)
    fig.update_layout(
        title=f"Faithful all-pairs M_h (L{args.layer}) — each edge in its own PCA-2 frame "
              f"(crimson=M_h spline, dashed=linear chord, faint=bootstrap refits)",
        margin=dict(l=20, r=40, t=70, b=20), height=700, width=1100)
    for k in range(len(keys)):
        fig.update_xaxes(title_text="mPC1 (value)", row=k // 3 + 1, col=k % 3 + 1)
        fig.update_yaxes(title_text="mPC2 (bend)", row=k // 3 + 1, col=k % 3 + 1)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"mh_simplex_faithful_L{args.layer}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
