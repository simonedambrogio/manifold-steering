"""Behavior slide: model value Q, true reward means, and aggregate performance.

A three-column Plotly panel for the presentation (cf. NHB Fig 1e/1f), built for a few
hand-picked example episodes where Llama clearly learns:

  col 1 — Model value estimate Q[t, arm]   (BestRL-recovered latent, one line per arm)
  col 2 — True reward mean   mu[t, arm]     (schedule's latent means, one line per arm)
  col 3 — Aggregate performance            (a single violin of per-episode relative
           reward over ALL episodes — fig1f, but pooled instead of split into blocks)

Columns 1-2 get one row per example episode; column 3 is a single panel spanning all
rows. Example episodes are auto-selected as the highest relative-reward episodes that
also show a restless best-arm switch (visible non-stationarity), or pass --episodes.

    .venv/bin/python -m figures.fig_behavior_panel \
        --q artifacts/value/bestrl_Q.pt \
        --dataset artifacts/datasets/llama8b_dataset.json \
        --out presentation/bandit/assets/behavior.html
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from analyses.behavior import episode_relative_rewards, load_logs
from bandit.schedule import generate_schedule

# tab10's first four — matches the matplotlib arm colors in fig_value_traces.
ARM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
HUMAN_COLOR = "#7b1fa2"   # purple: distinct from the arm palette, for the human reference dot


def _smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
    k = np.ones(w) / w
    return np.vstack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])]).T


def _leader_windows(M: np.ndarray, warm: int = 8, frac: float = 0.28):
    """Dominant arm and its lead margin in the early vs late window of a trace.

    Returns (early_leader, late_leader, margin_early, margin_late) where each margin is
    the average leader-minus-runner-up gap over the window, normalized by the trace range.
    """
    Ms = _smooth(M)
    a = int(len(Ms) * frac)
    early, late = Ms[warm:warm + a], Ms[-a:]
    rng = Ms.max() - Ms.min() + 1e-9

    def margin(win: np.ndarray) -> float:
        s = np.sort(win, axis=1)
        return float((s[:, -1] - s[:, -2]).mean() / rng)

    el = int(np.bincount(early.argmax(1), minlength=M.shape[1]).argmax())
    ll = int(np.bincount(late.argmax(1), minlength=M.shape[1]).argmax())
    return el, ll, margin(early), margin(late)


def find_crossover(Q, seeds, rr: np.ndarray, n_trials: int, scan: int = 160) -> int:
    """Best episode where the best arm changes durably in BOTH μ and Q (same swap).

    Among the top-`scan` learners (by relative reward), keep episodes whose early→late
    leader change agrees between the schedule means and the model's value estimate, then
    rank by the weakest of the four lead margins (clearest, most legible crossover first).
    """
    order = np.argsort(rr)[::-1]
    cands = []
    for ep in order[:scan]:
        mu = generate_schedule(seed=int(seeds[ep]), n_trials=n_trials).means
        emu, lmu, me_mu, ml_mu = _leader_windows(mu)
        eq, lq, me_q, ml_q = _leader_windows(Q[ep].numpy())
        if emu != lmu and (emu, lmu) == (eq, lq):           # same crossover in μ and Q
            cands.append((int(ep), min(me_mu, ml_mu, me_q, ml_q)))
    cands.sort(key=lambda c: -c[1])
    return cands[0][0] if cands else int(order[0])


def find_stable(rr: np.ndarray, seeds, n_trials: int, scan: int = 60) -> int:
    """Best learner that locks onto a single arm (no μ leader change) — a clean contrast."""
    order = np.argsort(rr)[::-1]
    for ep in order[:scan]:
        mu = generate_schedule(seed=int(seeds[ep]), n_trials=n_trials).means
        emu, lmu, me, ml = _leader_windows(mu)
        if emu == lmu and min(me, ml) > 0.12:               # one durable, clearly-ahead arm
            return int(ep)
    return int(order[0])


def pick_episodes(Q, rr: np.ndarray, seeds, n_trials: int) -> list[int]:
    """A stable single-arm learner followed by a clear best-arm crossover episode."""
    stable = find_stable(rr, seeds, n_trials)
    cross = find_crossover(Q, seeds, rr, n_trials)
    return [stable, cross]


# Shared layout constants so the two figures keep the proportions of the original
# combined panel: same per-row height, margins, and column aspect.
_ROW_H = 210          # px per example-episode row
_TOP, _BOT = 80, 95   # top/bottom margins (bottom holds the arm legend)


def build_traces_figure(Q, seeds, episodes: list[int], model_name: str) -> go.Figure:
    """Columns 1-2: model value estimate Q and true reward mean μ, one row per episode."""
    n_rows = len(episodes)
    n_arms, T = Q.shape[2], Q.shape[1]

    titles = ["Model value estimate (Q)", "True reward mean (μ)"] + [""] * (2 * n_rows - 2)
    fig = make_subplots(
        rows=n_rows, cols=2, column_widths=[0.5, 0.5],
        horizontal_spacing=0.08, vertical_spacing=0.07, subplot_titles=titles,
    )

    x = np.arange(T)
    for row, ep in enumerate(episodes, start=1):
        q_ep = Q[ep].numpy()
        mu = generate_schedule(seed=int(seeds[ep]), n_trials=T).means
        for i in range(n_arms):
            fig.add_trace(go.Scatter(
                x=x, y=q_ep[:, i], mode="lines", line=dict(color=ARM_COLORS[i], width=1.6),
                name=f"Arm {i + 1}", legendgroup=f"arm{i}", showlegend=(row == 1)), row, 1)
            fig.add_trace(go.Scatter(
                x=x, y=mu[:, i], mode="lines", line=dict(color=ARM_COLORS[i], width=1.6),
                legendgroup=f"arm{i}", showlegend=False), row, 2)
        fig.update_yaxes(title_text=f"ep {ep}", row=row, col=1)

    fig.update_xaxes(title_text="Trial", row=n_rows, col=1)
    fig.update_xaxes(title_text="Trial", row=n_rows, col=2)
    fig.update_layout(
        title=f"Model value estimate vs true reward means ({model_name})",
        width=800, height=_ROW_H * n_rows + 130, margin=dict(l=60, r=20, t=_TOP, b=_BOT),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_performance_figure(rr: np.ndarray, n_rows: int, model_name: str,
                             human_rr: float | None = None) -> go.Figure:
    """A single pooled violin of per-episode relative reward over all episodes.

    `n_rows` (the example count in the traces figure) only sets the height, so the violin
    keeps the same tall-narrow proportion it had spanning the combined panel. `human_rr`,
    if given, adds a differently-colored reference dot for the human study (read off NHB
    Fig 1f, which reports no exact number — the group means sit at ≈0.6).
    """
    fig = go.Figure()
    rng = np.random.default_rng(0)
    fig.add_trace(go.Violin(
        y=rr, x0=0, width=1.1, points=False, showlegend=False,
        fillcolor="#cccccc", line_color="#888888", opacity=0.7,
        meanline_visible=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=rng.uniform(-0.22, 0.22, size=len(rr)), y=rr, mode="markers", showlegend=False,
        marker=dict(size=4, color="#666666", opacity=0.45), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[0], y=[rr.mean()], mode="markers", name="Llama (group avg)",
        marker=dict(size=14, color="black")))
    if human_rr is not None:
        fig.add_trace(go.Scatter(
            x=[0], y=[human_rr], mode="markers", name="Humans (NHB Fig 1f, ≈0.6)",
            marker=dict(size=14, color=HUMAN_COLOR, line=dict(color="white", width=1.5))))

    fig.add_hline(y=1.0, line=dict(color="black", dash="dot", width=1))
    fig.add_hline(y=0.0, line=dict(color="black", dash="dash", width=1))
    fig.add_annotation(text="Optimal (impossible)", x=0.85, y=1.0, xref="x", yref="y",
                       xanchor="right", yanchor="top", showarrow=False, font=dict(size=11))
    fig.add_annotation(text="Random", x=0.85, y=0.0, xref="x", yref="y",
                       xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=11))

    fig.update_xaxes(range=[-0.9, 0.9], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="Relative reward", range=[min(rr.min(), 0) - 0.1, 1.12],
                     showgrid=True, gridcolor="#eee")
    fig.update_layout(
        title=f"Aggregate performance<br>({model_name}, {len(rr)} episodes)",
        width=360, height=_ROW_H * n_rows + 130, margin=dict(l=60, r=20, t=_TOP, b=_BOT),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        plot_bgcolor="white",
    )
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q", type=pathlib.Path, default=pathlib.Path("artifacts/value/bestrl_Q.pt"))
    p.add_argument("--dataset", type=pathlib.Path,
                   default=pathlib.Path("artifacts/datasets/llama8b_dataset.json"))
    p.add_argument("--outdir", type=pathlib.Path,
                   default=pathlib.Path("presentation/bandit/assets"))
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="episode indices to show (default: one stable learner + one crossover)")
    p.add_argument("--human-rr", type=float, default=0.60,
                   help="human reference relative reward (read off NHB Fig 1f; ≈0.6)")
    args = p.parse_args()

    data = torch.load(args.q, weights_only=False)
    Q, seeds = data["Q"], data["schedule_seeds"]
    model_name = "Llama-3.1-8B-Instruct"

    logs = load_logs(args.dataset)
    rr = episode_relative_rewards(logs)

    episodes = args.episodes or pick_episodes(Q, rr, seeds, Q.shape[1])
    print(f"example episodes: {episodes}  (relative reward "
          f"{', '.join(f'{rr[e]:+.2f}' for e in episodes)})")
    print(f"pooled relative reward: {rr.mean():+.3f} ± {rr.std():.3f}  (n={len(rr)})")

    args.outdir.mkdir(parents=True, exist_ok=True)
    traces = build_traces_figure(Q, seeds, episodes, model_name)
    perf = build_performance_figure(rr, len(episodes), model_name, human_rr=args.human_rr)
    for fig, name in [(traces, "behavior_traces.html"), (perf, "behavior_performance.html")]:
        out = args.outdir / name
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
