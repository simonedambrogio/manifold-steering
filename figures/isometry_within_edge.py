"""Within-edge isometry: the fair M_h↔M_y distance test (removes the skeleton-routing artifact).

For each edge we take M points along the fitted M_h/M_y splines and, over within-edge pairs,
compare two activation distances to the behavior (Hellinger) distance:
  manifold = arc-length along M_h     linear = straight Euclidean chord in PCA-64
Reports per-edge and pooled Pearson r, and plots D_activation vs D_behavior.

Two modes:
  default     read precomputed full-data pairs artifact (--pairs).
  trial-band  recompute the per-pair M_h/M_y on a trial window (--trial-min/--trial-max),
              e.g. 95–105, which removes trial-phase variance — testing whether the isometry
              is a value correspondence rather than a time-in-episode artifact.

    .venv/bin/python -m figures.isometry_within_edge --outdir bandit/figures            # full data
    .venv/bin/python -m figures.isometry_within_edge --trial-min 95 --trial-max 106     # band
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
from scipy.stats import pearsonr

from analyses.fit_behavior_manifold import exp_map, log_map
from analyses.fit_pair_manifold import _bin_centroids, _fit_splines, _quantile_edges

EDGE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def _hell(a, b):
    return np.linalg.norm(np.sqrt(a) - np.sqrt(b)) / np.sqrt(2)


def _edges_from_artifact(args):
    d = torch.load(args.pairs, weights_only=False)
    letters = d["letters"]
    edges = [(f"{letters[pd['i']]}{letters[pd['j']]}", pd["manifold_grid"].numpy(), pd["dist_grid"].numpy())
             for pd in d["pairs"].values()]
    return edges, "all trials", ""


def _edges_from_band(args):
    """Recompute per-pair M_h (PCA-64 spline) and M_y (Hellinger spline) on a trial window."""
    act = torch.load(args.activations, weights_only=False)
    ep = act["episode_ids"]; A_all = act["activations"]
    n_eps, T, _, hid = A_all.shape
    tmin = max(1, args.trial_min); tmax = args.trial_max if args.trial_max > 0 else T
    A = A_all[:, tmin:tmax, args.layer, :].float().reshape(-1, hid); del A_all, act
    Q = torch.load(args.q, weights_only=False)["Q"][ep].cpu().numpy(); na = Q.shape[2]
    logs = json.loads(args.dataset.read_text()); letters = logs[0]["letters"]
    ch = np.array([l["choice_dists"] for l in logs])[:, tmin:tmax, :na]
    ch = ch / ch.sum(-1, keepdims=True)
    mu = A.mean(0); _, _, Vh = torch.linalg.svd(A - mu, full_matrices=False)
    Xp = ((A - mu) @ Vh[:64].T).cpu().numpy()
    Qp = Q[:, tmin:tmax].reshape(-1, na); Cp = ch.reshape(-1, na)
    order = np.argsort(-Qp, axis=1)
    edges = []
    for i, j in itertools.combinations(range(na), 2):
        sel = (np.sort(order[:, :2], axis=1) == np.array([i, j])).all(axis=1)
        if sel.sum() < args.n_bins * 3:
            continue
        r = (Qp[sel, j] - Qp[sel, i]).astype(float)
        e = _quantile_edges(r, args.n_bins)
        cenh, rc, cnt = _bin_centroids(r, Xp[sel], e)
        w = np.sqrt(cnt.astype(float))
        _, g = _fit_splines(rc, cenh, w, float(len(rc)), args.n_grid)
        cend, _, _ = _bin_centroids(r, Cp[sel], e)
        sq = np.sqrt(cend); b = sq.mean(0); b /= np.linalg.norm(b)
        _, tg = _fit_splines(rc, log_map(b, sq), w, float(len(rc)), args.n_grid)
        dy = exp_map(b, tg) ** 2; dy /= dy.sum(1, keepdims=True)
        edges.append((f"{letters[i]}{letters[j]}", g, dy))
    return edges, f"trials {tmin}–{tmax-1} (trial-phase removed)", f"_t{tmin}-{tmax-1}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=pathlib.Path,
                   default=pathlib.Path("artifacts/manifolds/llama8b_fixed_pairs_L19.pt"))
    p.add_argument("--activations", type=pathlib.Path,
                   default=pathlib.Path("artifacts/activations/llama8b_fixed_residual.pt"))
    p.add_argument("--q", type=pathlib.Path, default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("artifacts/datasets/llama8b_fixed.json"))
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--trial-min", type=int, default=0)
    p.add_argument("--trial-max", type=int, default=0, help=">0 enables trial-band recompute mode")
    p.add_argument("--n-bins", type=int, default=12)
    p.add_argument("--n-grid", type=int, default=150)
    p.add_argument("--m-points", type=int, default=15)
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("bandit/figures"))
    args = p.parse_args()

    band = args.trial_max > 0
    edges, rng_txt, suffix = _edges_from_band(args) if band else _edges_from_artifact(args)

    per_edge = {}
    man_all, lin_all, y_all, col_all = [], [], [], []
    for k, (name, g, dy) in enumerate(edges):
        cumarc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))])
        sub = np.linspace(0, len(g) - 1, args.m_points).round().astype(int)
        man, lin, yv = [], [], []
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                ia, ib = sub[a], sub[b]
                man.append(abs(cumarc[ib] - cumarc[ia]))
                lin.append(np.linalg.norm(g[ia] - g[ib]))
                yv.append(_hell(dy[ia], dy[ib]))
        man, lin, yv = map(np.array, (man, lin, yv))
        per_edge[name] = (pearsonr(man, yv)[0], pearsonr(lin, yv)[0])
        man_all += man.tolist(); lin_all += lin.tolist(); y_all += yv.tolist()
        col_all += [EDGE_COLORS[k % len(EDGE_COLORS)]] * len(man)

    man_all, lin_all, y_all = map(np.array, (man_all, lin_all, y_all))
    R_man = pearsonr(man_all, y_all)[0]; R_lin = pearsonr(lin_all, y_all)[0]
    rm = np.mean([v[0] for v in per_edge.values()]); rl = np.mean([v[1] for v in per_edge.values()])

    print(f"[{rng_txt}] per-edge Pearson r (manifold, linear):")
    for nm, (a, b) in per_edge.items():
        print(f"  {nm}: manifold {a:.3f}   linear {b:.3f}")
    print(f"mean per-edge: manifold {rm:.3f}  linear {rl:.3f}")
    print(f"pooled:        manifold {R_man:.3f}  linear {R_lin:.3f}")

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, subplot_titles=(
        f"Manifold (arc-length) vs behavior   pooled r={R_man:.3f}",
        f"Linear (chord) vs behavior   pooled r={R_lin:.3f}"))
    fig.add_trace(go.Scatter(x=man_all, y=y_all, mode="markers", showlegend=False,
        marker=dict(size=4, color=col_all, opacity=0.5)), 1, 1)
    fig.add_trace(go.Scatter(x=lin_all, y=y_all, mode="markers", showlegend=False,
        marker=dict(size=4, color=col_all, opacity=0.5)), 1, 2)
    fig.update_xaxes(title_text="activation distance (manifold arc)", row=1, col=1)
    fig.update_xaxes(title_text="activation distance (linear chord)", row=1, col=2)
    fig.update_yaxes(title_text="behavior distance (Hellinger)", row=1, col=1)
    fig.update_yaxes(title_text="behavior distance (Hellinger)", row=1, col=2)
    fig.update_layout(
        title=f"Within-edge isometry — {rng_txt} (L{args.layer}) — "
              f"mean per-edge: manifold r={rm:.3f}, linear r={rl:.3f}  (colored by edge)",
        height=480, width=1000, margin=dict(l=60, r=20, t=80, b=50))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"isometry_within_edge_L{args.layer}{suffix}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
