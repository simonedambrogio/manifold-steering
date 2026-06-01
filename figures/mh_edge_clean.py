"""Refined single-edge M_h picture: manifold-aligned projection + bootstrap noise band.

Improves on `figures.mh_edge` for *presentation*: (1) projects onto the manifold's own
axes (PCA-3 of the fitted spline) instead of the pair-cloud's top-3, so the curve isn't
tilted through nuisance variance; (2) overlays an episode-bootstrap band (resample episodes,
refit) so the spline's noise-sensitivity is visible, not guessed; (3) a smoothing dropdown
shows the wobble-vs-oversmoothing tradeoff. Cloud at low opacity. Geometry stays 64-D.

    .venv/bin/python -m figures.mh_edge_clean --layer 19 --pair 0_1 --outdir bandit/figures
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch

from analyses.fit_pair_manifold import _bin_centroids, _fit_splines, _quantile_edges


def _spline_grid(r, X, edges, s, n_grid):
    cen, rc, cnt = _bin_centroids(r, X, edges)
    w = np.sqrt(cnt.astype(float))
    _, grid = _fit_splines(rc, cen, w, s, n_grid)
    return grid, cen, rc, cnt


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--activations", type=pathlib.Path,
                   default=pathlib.Path("artifacts/activations/llama8b_fixed_residual.pt"))
    p.add_argument("--q", type=pathlib.Path,
                   default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--dataset", type=pathlib.Path,
                   default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--pair", default="0_1")
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=25)
    p.add_argument("--n-grid", type=int, default=150)
    p.add_argument("--n-boot", type=int, default=30)
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()
    i, j = (int(x) for x in args.pair.split("_"))

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
    ep_idx = np.repeat(np.arange(n_eps), T - 1)
    sel = (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)
    r = (Qp[sel, j] - Qp[sel, i]).astype(np.float64)
    X = A_pca[sel]; eps = ep_idx[sel]
    print(f"edge {letters[i]}-{letters[j]} (L{args.layer}): n={len(r)}")

    edges = _quantile_edges(r, args.n_bins)
    cen, rc, cnt = _bin_centroids(r, X, edges)
    n_used = len(rc)
    grid0, cen, rc, cnt = _spline_grid(r, X, edges, float(n_used), args.n_grid)

    # manifold-aligned projection: PCA-3 of the fitted spline grid
    mu3 = grid0.mean(0)
    _, S3, V3 = np.linalg.svd(grid0 - mu3, full_matrices=False)
    W3 = V3[:3]; var3 = (S3[:3] ** 2) / (S3 ** 2 + 1e-9).sum()
    to3 = lambda M: (M - mu3) @ W3.T
    print(f"manifold-aligned PCA-3 var of the spline: {var3.round(3).tolist()} (sum {var3.sum():.2f})")

    # bootstrap band: resample episodes, refit, project with the FIXED frame
    ue = np.unique(eps); rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.n_boot):
        samp = rng.choice(ue, size=len(ue), replace=True)
        idx = np.concatenate([np.where(eps == e)[0] for e in samp])
        rb, Xb = r[idx], X[idx]
        gb, *_ = _spline_grid(rb, Xb, _quantile_edges(rb, args.n_bins), float(n_used), args.n_grid)
        boot.append(to3(gb))
    boot = np.stack(boot)  # [B, n_grid, 3]
    # quantify noise: mean pointwise bootstrap spread vs the curve's own extent (3D)
    band_w = np.linalg.norm(boot.std(0), axis=1).mean()         # mean pointwise sd in 3D
    arc = np.linalg.norm(np.diff(to3(grid0), axis=0), axis=1).sum()
    print(f"bootstrap band width ≈ {band_w:.3f}; curve arc (3D) ≈ {arc:.2f}; "
          f"band/arc ≈ {band_w/arc:.1%}")

    P = to3(X); G0 = to3(grid0); C = to3(cen)
    lo = X[r <= np.quantile(r, 0.1)].mean(0); hi = X[r >= np.quantile(r, 0.9)].mean(0)
    chord = to3(np.stack([lo, hi]))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="trials",
        marker=dict(size=2.2, color=r, colorscale="Turbo", opacity=0.10,
                    colorbar=dict(title=f"r=Q[{letters[j]}]−Q[{letters[i]}]"))))
    # bootstrap replicates (faint) — the noise envelope
    for b in range(args.n_boot):
        fig.add_trace(go.Scatter3d(x=boot[b, :, 0], y=boot[b, :, 1], z=boot[b, :, 2], mode="lines",
            line=dict(color="rgba(220,20,60,0.12)", width=2), showlegend=(b == 0),
            name="bootstrap refits", legendgroup="boot", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=chord[:, 0], y=chord[:, 1], z=chord[:, 2], mode="lines",
        name="linear chord", line=dict(color="black", width=4, dash="dash")))
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", name="centroids",
        marker=dict(size=4 + 9 * np.sqrt(cnt / cnt.max()), color=rc, colorscale="Turbo",
                    line=dict(color="white", width=1))))
    # main spline at several smoothing levels (dropdown)
    s_levels = [("mild (s=n)", float(n_used)), ("medium (5n)", 5.0 * n_used),
                ("heavy (20n)", 20.0 * n_used)]
    spline_traces = []
    for k, (lbl, s) in enumerate(s_levels):
        g, *_ = _spline_grid(r, X, edges, s, args.n_grid); g3 = to3(g)
        fig.add_trace(go.Scatter3d(x=g3[:, 0], y=g3[:, 1], z=g3[:, 2], mode="lines",
            name=f"M_h spline [{lbl}]", visible=(k == 0), line=dict(color="crimson", width=7)))
        spline_traces.append(len(fig.data) - 1)

    n_fixed = spline_traces[0]  # index of first spline trace
    def vis(k):
        v = [True] * n_fixed
        v += [m == k for m in range(len(s_levels))]
        return v
    buttons = [dict(label=lbl, method="update", args=[{"visible": vis(k)}])
               for k, (lbl, _) in enumerate(s_levels)]
    fig.update_layout(
        title=f"M_h edge {letters[i]}↔{letters[j]} (L{args.layer}) — manifold-aligned PCA-3 "
              f"(var {var3.sum():.0%}); bootstrap band/arc ≈ {band_w/arc:.0%}",
        scene=dict(xaxis_title="mPC1", yaxis_title="mPC2", zaxis_title="mPC3"),
        updatemenus=[dict(buttons=buttons, x=0, y=1, xanchor="left", yanchor="top", showactive=True)],
        margin=dict(l=0, r=0, t=60, b=0), legend=dict(x=0, y=0))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"mh_edge_{letters[i]}{letters[j]}_L{args.layer}_clean.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
