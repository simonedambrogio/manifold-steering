"""Decode every arm's value from Llama's residual stream (fixed-letter data, L19).

On the **fixed-letter** deck (arms always labelled B, C, D, F), a ridge probe fit from the
decision-token activation recovers each arm's BestRL value — not only the best arm. We
report **held-out decodability** as the squared Pearson correlation between decoded and
true value on episodes never seen in training (a metric robust to the −222 value floor that
otherwise wrecks absolute-scale R²; see dev/context/value-steering/report_experiment.md).

Top row: decoded vs. true value, one panel per arm. Bottom: decodability by value rank
(best → worst), showing the graded structure — best dominant, every arm above chance.

    .venv/bin/python -m figures.fig_decode_arms \
        --activations artifacts/activations/llama8b_fixed_residual.pt \
        --q artifacts/value/bestrl_Q_fixed.pt \
        --out bandit/figures/decode_arms.html
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

EPS = 1e-9
ARM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]   # tab10 0-3, as in fig_behavior_panel
RANK_COLORS = ["#333333", "#555555", "#777777", "#999999"]   # best → worst (dark → mid grey)
LETTERS = ["B", "C", "D", "F"]                               # fixed arm labels on the deck
RANKS = ["best", "2nd", "3rd", "worst"]


def _ridge(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    p = X.shape[1]
    return torch.linalg.solve(X.T @ X + alpha * torch.eye(p), X.T @ Y)


def _corr2(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Squared Pearson correlation between a predicted and a true column."""
    p = pred - pred.mean()
    t = y - y.mean()
    return float((p @ t) ** 2 / ((p @ p) * (t @ t) + EPS))


def decode(activations, Q, layer: int, alphas, seed: int = 0):
    """Per-arm and per-rank held-out decodability + held-out scatter points at one layer."""
    A = activations[:, :, layer, :].float()                 # [E, T, H]
    sorted_q, _ = Q.sort(dim=-1, descending=True)
    # Y columns: 4 per-arm (B,C,D,F) then 4 by-rank (best,2nd,3rd,worst).
    Y = torch.cat([Q, sorted_q], dim=-1)                    # [E, T, 8]
    names = LETTERS + RANKS

    A, Y = A[:, 1:], Y[:, 1:]                                # drop trial 0 (uninformative init)
    E = A.shape[0]
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(seed))
    n_tr, n_va = int(round(0.8 * E)), int(round(0.1 * E))
    i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    flat = lambda t, idx: t[idx].reshape(-1, t.shape[-1])
    Xtr, Xva, Xte = flat(A, i_tr), flat(A, i_va), flat(A, i_te)
    Ytr, Yva, Yte = flat(Y, i_tr), flat(Y, i_va), flat(Y, i_te)

    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    # Per-target ridge: pick α by validation corr², keep that α's held-out prediction.
    fits = [( _ridge(Xtr, Ytr, a),) for a in alphas]
    betas = [f[0] for f in fits]
    Pva = [Xva @ b for b in betas]
    Pte = [Xte @ b for b in betas]
    result = {"true": Yte, "pred": torch.empty_like(Yte), "names": names,
              "corr2": {}, "alpha": {}, "n_test_eps": int(len(i_te))}
    for j, name in enumerate(names):
        k = int(np.argmax([_corr2(Pva[a][:, j], Yva[:, j]) for a in range(len(alphas))]))
        result["pred"][:, j] = Pte[k][:, j]
        result["corr2"][name] = _corr2(Pte[k][:, j], Yte[:, j])
        result["alpha"][name] = float(alphas[k])
    return result


def _scatter_panel(fig, x, y, color, corr2, row, col, axis_idx, max_pts, rng):
    """One decoded-vs-true panel: points, best-fit trend, and an r² annotation.

    Each axis is scaled to its OWN data. Decodability here is association (correlation),
    and a shared square x/y range mis-frames targets whose decoded and true values sit in
    disjoint bands (e.g. the floored worst arm: true ≈ −200, decoded ≈ 0).
    """
    if len(x) > max_pts:
        sel = rng.choice(len(x), max_pts, replace=False)
        x, y = x[sel], y[sel]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", showlegend=False,
        marker=dict(size=3, color=color, opacity=0.28), hoverinfo="skip"), row, col)
    # best-fit line: the fitted trend the points follow, not the y=x identity.
    slope = float(np.cov(x, y)[0, 1] / (x.var() + EPS))
    icpt = float(y.mean() - slope * x.mean())
    xlo, xhi = float(x.min()), float(x.max())
    xpad = 0.05 * (xhi - xlo)
    xl = np.array([xlo - xpad, xhi + xpad])
    yl = slope * xl + icpt
    fig.add_trace(go.Scatter(
        x=xl, y=yl, mode="lines",
        line=dict(color="#444444", width=1.2), showlegend=False, hoverinfo="skip"), row, col)
    ylo = min(float(y.min()), float(yl.min())); yhi = max(float(y.max()), float(yl.max()))
    ypad = 0.05 * (yhi - ylo)
    fig.add_annotation(text=f"r²={corr2:.2f}", xref=f"x{axis_idx}", yref=f"y{axis_idx}",
                       x=xlo - xpad, y=yhi + ypad, xanchor="left", yanchor="top",
                       showarrow=False, font=dict(size=13, color=color), row=row, col=col)
    fig.update_xaxes(range=[xlo - xpad, xhi + xpad], row=row, col=col)
    fig.update_yaxes(range=[ylo - ypad, yhi + ypad], row=row, col=col)


def build_figure(res: dict, layer: int, max_pts: int = 2000, seed: int = 0) -> go.Figure:
    true, pred = res["true"].numpy(), res["pred"].numpy()
    rng = np.random.default_rng(seed)

    rank_titles = ["Best", "2nd", "3rd", "Worst"]
    fig = make_subplots(
        rows=2, cols=4, vertical_spacing=0.13, horizontal_spacing=0.05,
        subplot_titles=[f"Arm {L}" for L in LETTERS] + rank_titles,
    )

    # Row 1 — decode by arm identity (B,C,D,F); Row 2 — decode by value rank.
    # res columns are [B,C,D,F, best,2nd,3rd,worst]; ranks are columns 4-7.
    for j, (L, color) in enumerate(zip(LETTERS, ARM_COLORS)):
        _scatter_panel(fig, true[:, j], pred[:, j], color, res["corr2"][L],
                       row=1, col=j + 1, axis_idx=j + 1, max_pts=max_pts, rng=rng)
    for j, (rk, color) in enumerate(zip(RANKS, RANK_COLORS)):
        _scatter_panel(fig, true[:, 4 + j], pred[:, 4 + j], color, res["corr2"][rk],
                       row=2, col=j + 1, axis_idx=5 + j, max_pts=max_pts, rng=rng)

    for r in (1, 2):
        fig.update_yaxes(title_text="decoded value", row=r, col=1)
    fig.update_xaxes(title_text="true value", row=2, col=1)

    fig.update_layout(
        title="Decoding arm values from Llama's residual stream (fixed letters, layer "
              f"{layer})<br><sup>top row: each arm by identity (B,C,D,F) &nbsp;·&nbsp; "
              "bottom row: by value rank (best → worst)</sup>",
        width=1000, height=660, margin=dict(l=60, r=20, t=110, b=55),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", type=pathlib.Path,
                   default=pathlib.Path("artifacts/activations/llama8b_fixed_residual.pt"))
    p.add_argument("--q", type=pathlib.Path,
                   default=pathlib.Path("artifacts/value/bestrl_Q_fixed.pt"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bandit/figures/decode_arms.html"))
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0])
    args = p.parse_args()

    act = torch.load(args.activations, weights_only=False, mmap=True)
    ep_ids = list(act["episode_ids"])
    q = torch.load(args.q, weights_only=False)
    Q = q["Q"][ep_ids].float()                              # [E, T, 4], arm index = stable letter

    res = decode(act["activations"], Q, args.layer, args.alphas)
    print(f"held-out decodability (squared correlation) at L{args.layer}, "
          f"{res['n_test_eps']} test episodes:")
    print("  per arm:  " + "  ".join(f"{L}={res['corr2'][L]:.3f}" for L in LETTERS))
    print("  by rank:  " + "  ".join(f"{r}={res['corr2'][r]:.3f}" for r in RANKS))

    fig = build_figure(res, args.layer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"saved → {args.out}")

    summary = args.out.with_suffix(".json")
    summary.write_text(json.dumps({
        "layer": args.layer, "metric": "held-out squared Pearson correlation",
        "n_test_episodes": res["n_test_eps"], "letters": LETTERS,
        "corr2": res["corr2"], "alpha": res["alpha"],
    }, indent=2))
    print(f"saved → {summary}")


if __name__ == "__main__":
    main()
