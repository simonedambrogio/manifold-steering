"""M_h activation-manifold plot for one value-transition edge (bandit choice version).

Renders the activation manifold M_h for a single arm pair (i, j), restricted to trials
where {i, j} are the top-2 arms, with intrinsic coordinate r = Q[j] - Q[i] (the signed
best-vs-second margin; r=0 is the decision boundary). Mirrors `analyses.fit_pair_manifold`
exactly (same PCA-64 subspace, quantile bins, sqrt-count-weighted cubic splines) so the
picture is faithful to the steering pipeline.

The figure is a PCA-3 *illustration only* — curvature/isometry are measured in PCA-64,
never read off the 3D. Built-in confound controls:
  * n_bins dropdown {12,20,30,40}      — is the manifold shape stable, or a binning artifact?
  * color-mode toggle {r, trial#, recency-gap} — does the value-ordering differ from the
    time/recency-ordering? (if recoloring by trial# / last-reward reproduces the order,
    the geometry is confounded with time/recency rather than value.)
A companion checks panel plots, per r-bin: count, mean trial#, mean recency-gap, mean level
(Q[i]+Q[j])/2 — to verify the margin axis isn't carrying trial-phase / recency / abs-level.

    .venv/bin/python -m figures.mh_edge \
        --activations artifacts/activations/llama8b_fixed_residual.pt \
        --q artifacts/value/bestrl_Q_fixed.pt \
        --dataset artifacts/datasets/llama8b_fixed.json \
        --layer 19 --outdir bandit/figures
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


def _last_reward(actions: np.ndarray, rewards: np.ndarray, n_arms: int) -> np.ndarray:
    """[n_eps, T, n_arms]: most recent reward seen for each arm *before* trial t (nan if none)."""
    n_eps, T = actions.shape
    out = np.full((n_eps, T, n_arms), np.nan)
    for e in range(n_eps):
        seen = [np.nan] * n_arms
        for t in range(T):
            out[e, t] = seen
            seen = list(seen)
            seen[int(actions[e, t])] = float(rewards[e, t])
    return out


def _pca3(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA-3 on X [n, d]; return (mean [d], W [3, d], var_explained [3])."""
    mu = X.mean(axis=0)
    _, S, Vh = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vh[:3], (S[:3] ** 2) / (S ** 2).sum()


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
    p.add_argument("--pair", default=None, help="'i_j' arm indices; default = max top-2 count")
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--n-bins-list", default="12,20,30,40")
    p.add_argument("--n-grid", type=int, default=150)
    p.add_argument("--check-bins", type=int, default=30, help="n_bins for the confound panel")
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    # ---- load, mirroring fit_pair_manifold's alignment ----
    act = torch.load(args.activations, weights_only=False)
    ep_ids = act["episode_ids"]
    activations = act["activations"]
    n_eps, T, _, hidden = activations.shape
    A = activations[:, 1:, args.layer, :].float().reshape(-1, hidden)   # [N, hidden], skip t0
    del activations, act

    qd = torch.load(args.q, weights_only=False)
    Q = qd["Q"][ep_ids].cpu().numpy()
    n_arms = Q.shape[2]
    actions = qd["actions"][ep_ids].cpu().numpy()
    rewards = qd["rewards"][ep_ids].cpu().numpy()

    logs = [json.loads(args.dataset.read_text())[e] for e in ep_ids]
    letters = logs[0]["letters"]

    # PCA-64 (global subspace, paper-style), then per-trial views aligned to A's rows
    A_mean = A.mean(dim=0)
    _, _, Vh = torch.linalg.svd(A - A_mean, full_matrices=False)
    pca = Vh[: args.pca_dim]
    A_pca = ((A - A_mean) @ pca.T).cpu().numpy()                         # [N, 64]
    Qp = Q[:, 1:].reshape(-1, n_arms)
    order = np.argsort(-Qp, axis=1)
    trial_idx = np.tile(np.arange(1, T), n_eps)
    lastr = _last_reward(actions, rewards, n_arms)[:, 1:].reshape(-1, n_arms)

    # ---- choose pair (max top-2 count unless given) ----
    def sel_of(i, j):
        return (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)

    if args.pair:
        i, j = (int(x) for x in args.pair.split("_"))
    else:
        counts = {(i, j): int(sel_of(i, j).sum()) for i, j in itertools.combinations(range(n_arms), 2)}
        (i, j) = max(counts, key=counts.get)
        print("top-2 counts per pair:", {f"{letters[a]}-{letters[b]}": c for (a, b), c in counts.items()})

    sel = sel_of(i, j)
    r = (Qp[sel, j] - Qp[sel, i]).astype(np.float64)
    A_pair = A_pca[sel]                                                  # [n, 64]
    level = (Qp[sel, i] + Qp[sel, j]) / 2.0
    rec_gap = lastr[sel, j] - lastr[sel, i]
    tr = trial_idx[sel].astype(float)
    print(f"pair {letters[i]}-{letters[j]}: n={sel.sum()} top-2 trials; r∈[{r.min():.0f},{r.max():.0f}]")

    # ---- fixed 3D projection on the pair-restricted PCA-64 cloud ----
    mu3, W3, var3 = _pca3(A_pair)
    to3 = lambda X64: (X64 - mu3) @ W3.T
    P = to3(A_pair)                                                      # [n, 3]
    print(f"PCA-3 var explained (within pair): {var3.round(3).tolist()} (sum {var3.sum():.2f})")

    # confound correlations (the numbers behind the panel)
    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")
    print(f"corr(r, trial#)   = {corr(r, tr):+.2f}")
    print(f"corr(r, recency-gap)= {corr(r, rec_gap):+.2f}")
    print(f"corr(r, level)    = {corr(r, level):+.2f}")

    # ---- manifold per n_bins ----
    n_bins_list = [int(x) for x in args.n_bins_list.split(",")]

    def manifold(nb):
        edges = _quantile_edges(r, nb)
        cen, rc, cnt = _bin_centroids(r, A_pair, edges)
        w = np.sqrt(cnt.astype(float))
        rg, grid = _fit_splines(rc, cen, w, float(len(rc)), args.n_grid)
        return to3(cen), rc, cnt, to3(grid), rg

    # chord: line between bottom-decile and top-decile mean activation (nb-independent)
    lo = A_pair[r <= np.quantile(r, 0.1)].mean(0)
    hi = A_pair[r >= np.quantile(r, 0.9)].mean(0)
    chord = to3(np.stack([lo, hi]))

    # ---- main 3D figure ----
    fig = go.Figure()
    colorscale = "Turbo"
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="trials",
        marker=dict(size=2.5, color=r, colorscale=colorscale, opacity=0.12,
                    colorbar=dict(title="r = Q[%s]−Q[%s]" % (letters[j], letters[i]), x=1.02)),
        customdata=np.stack([r, tr, rec_gap, level], axis=1),
        hovertemplate="r=%{customdata[0]:.1f}<br>trial=%{customdata[1]:.0f}<br>"
                      "rec-gap=%{customdata[2]:.1f}<br>level=%{customdata[3]:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=chord[:, 0], y=chord[:, 1], z=chord[:, 2], mode="lines", name="linear chord",
        line=dict(color="black", width=4, dash="dash"),
    ))
    vis_blocks = []
    for nb in n_bins_list:
        cen3, rc, cnt, grid3, rg = manifold(nb)
        fig.add_trace(go.Scatter3d(
            x=cen3[:, 0], y=cen3[:, 1], z=cen3[:, 2], mode="markers", visible=(nb == n_bins_list[0]),
            name=f"centroids (n_bins={nb})",
            marker=dict(size=4 + 9 * np.sqrt(cnt / cnt.max()), color=rc, colorscale=colorscale,
                        line=dict(color="white", width=1)),
            customdata=np.stack([rc, cnt], axis=1),
            hovertemplate="r=%{customdata[0]:.1f}<br>count=%{customdata[1]:.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=grid3[:, 0], y=grid3[:, 1], z=grid3[:, 2], mode="lines", visible=(nb == n_bins_list[0]),
            name=f"M_h spline (n_bins={nb})", line=dict(color="crimson", width=6),
        ))
        vis_blocks.append(nb)

    base = 2  # scatter + chord always on
    def vis_for(k):
        v = [True, True]
        for m in range(len(vis_blocks)):
            v += [m == k, m == k]
        return v

    bin_buttons = [dict(label=f"n_bins={nb}", method="update", args=[{"visible": vis_for(k)}])
                   for k, nb in enumerate(vis_blocks)]
    color_buttons = [
        dict(label="color: r (value)", method="restyle",
             args=[{"marker.color": [r], "marker.colorbar.title": "r (value gap)"}, [0]]),
        dict(label="color: trial #", method="restyle",
             args=[{"marker.color": [tr], "marker.colorbar.title": "trial #"}, [0]]),
        dict(label="color: recency-gap", method="restyle",
             args=[{"marker.color": [rec_gap], "marker.colorbar.title": "last-reward gap"}, [0]]),
    ]
    fig.update_layout(
        title=f"M_h edge {letters[i]}↔{letters[j]} (L{args.layer}) — PCA-3 of pair-restricted "
              f"PCA-{args.pca_dim} cloud (var {var3.sum():.0%}); geometry measured in {args.pca_dim}-D",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        updatemenus=[
            dict(buttons=bin_buttons, x=0.0, y=1.0, xanchor="left", yanchor="top", showactive=True),
            dict(buttons=color_buttons, x=0.0, y=0.88, xanchor="left", yanchor="top", showactive=True),
        ],
        legend=dict(x=0.0, y=0.0), margin=dict(l=0, r=0, t=60, b=0),
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    main_path = args.outdir / f"mh_edge_{letters[i]}{letters[j]}_L{args.layer}.html"
    fig.write_html(main_path, include_plotlyjs="cdn")
    print(f"saved → {main_path}")

    # ---- confound checks panel ----
    edges = _quantile_edges(r, args.check_bins)
    def binmean(x):
        c, rc, _ = _bin_centroids(r, np.asarray(x, float)[:, None], edges)
        return rc, c[:, 0]
    _, rc_c, counts = _bin_centroids(r, np.ones((len(r), 1)), edges)
    chk = make_subplots(rows=2, cols=2, subplot_titles=(
        "bin count vs r", "mean trial # vs r", "mean recency-gap vs r", "mean level vs r"))
    chk.add_trace(go.Scatter(x=rc_c, y=counts, mode="lines+markers"), 1, 1)
    rc, yt = binmean(tr);                   chk.add_trace(go.Scatter(x=rc, y=yt, mode="lines+markers"), 1, 2)
    rc, yg = binmean(np.nan_to_num(rec_gap, nan=np.nanmean(rec_gap))); chk.add_trace(go.Scatter(x=rc, y=yg, mode="lines+markers"), 2, 1)
    rc, yl = binmean(level);                chk.add_trace(go.Scatter(x=rc, y=yl, mode="lines+markers"), 2, 2)
    chk.update_layout(showlegend=False, title=f"Confound checks — edge {letters[i]}↔{letters[j]} "
                      f"(L{args.layer}); r should NOT track trial#/recency/level monotonically")
    for k in range(1, 5):
        chk.update_xaxes(title_text="r", row=(k - 1) // 2 + 1, col=(k - 1) % 2 + 1)
    chk_path = args.outdir / f"mh_edge_{letters[i]}{letters[j]}_L{args.layer}_checks.html"
    chk.write_html(chk_path, include_plotlyjs="cdn")
    print(f"saved → {chk_path}")


if __name__ == "__main__":
    main()
