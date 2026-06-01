"""Model-comparison slide (presentation): the action-prediction panel only.

A single Plotly panel mirroring the matplotlib right panel of
`fig_model_comparison.py` (NHB Fig 2d style) and styled to match the slide-3
performance figure in `fig_behavior_panel.py` (360x550, white background, light
grid, no title). One point per model with SEM error bars over the 40 test
episodes, a chance reference line, and significance brackets between adjacent
flexible models from paired t-tests.

    .venv/bin/python -m figures.fig_model_comparison_panel \
        --fits-dir artifacts/fits/ \
        --out presentation/bandit/assets/model_comparison.html
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Cognitive ladder, simplest -> most flexible (same order as fig_model_comparison).
MODEL_ORDER = ["SimpleRL", "BestRL", "VanillaRNN", "MemoryANN"]
CHANCE = 0.25  # uniform policy over 4 arms

# Slide-3 figure proportions (see fig_behavior_panel.build_performance_figure).
_W, _H = 360, 550


def _p_label(p: float) -> str:
    if p < 0.001:
        return "P < 0.001"
    if p < 0.05:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"


def _load_fits(fits_dir: pathlib.Path) -> list[dict]:
    fits = {}
    for p in sorted(fits_dir.glob("*.json")):
        d = json.loads(p.read_text())
        name = d["model_name"]
        # Guard against stale duplicate files (e.g. an old simple_rl.json alongside
        # simplerl.json): keep whichever carries the per-episode data we plot.
        if len(d.get("test_per_episode_acc", [])) < len(
            fits.get(name, {}).get("test_per_episode_acc", [])
        ):
            continue
        fits[name] = d
    return [fits[name] for name in MODEL_ORDER if name in fits]


def build_accuracy_figure(fits: list[dict]) -> go.Figure:
    names = [f["model_name"] for f in fits]
    per_eps = [np.asarray(f.get("test_per_episode_acc", []), dtype=float) for f in fits]
    means = [float(a.mean()) for a in per_eps]
    sems = [float(a.std(ddof=1) / np.sqrt(len(a))) for a in per_eps]
    x = list(range(len(names)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=means,
        error_y=dict(type="data", array=sems, visible=True, thickness=1.3,
                     width=5, color="black"),
        mode="markers", marker=dict(size=11, color="black"),
        showlegend=False, hoverinfo="skip"))

    # Chance reference line + label.
    fig.add_hline(y=CHANCE, line=dict(color="gray", dash="dash", width=1))
    fig.add_annotation(text="chance", x=len(names) - 0.6, y=CHANCE, xref="x", yref="y",
                       xanchor="right", yanchor="bottom", showarrow=False,
                       font=dict(size=11, color="gray"))

    # Significance brackets between adjacent flexible models (paired t-test).
    def bracket(i: int, j: int, y: float) -> None:
        t, p = stats.ttest_rel(per_eps[i], per_eps[j])
        h = 0.008
        fig.add_trace(go.Scatter(
            x=[i, i, j, j], y=[y, y + h, y + h, y], mode="lines",
            line=dict(color="black", width=0.9), showlegend=False, hoverinfo="skip"))
        fig.add_annotation(text=_p_label(float(p)), x=(i + j) / 2, y=y + h, yshift=7,
                           xref="x", yref="y", showarrow=False, font=dict(size=11))

    y_top = max(m + s for m, s in zip(means, sems))
    bracket(1, 2, y_top + 0.012)   # BestRL vs VanillaRNN
    bracket(2, 3, y_top + 0.033)   # VanillaRNN vs MemoryANN

    fig.update_xaxes(tickvals=x, ticktext=names, tickangle=-30, tickfont=dict(size=11),
                     range=[-0.5, len(names) - 0.4], showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="Action accuracy", range=[0.22, 0.56],
                     showgrid=True, gridcolor="#eee")
    fig.update_layout(
        width=_W, height=_H, margin=dict(l=55, r=20, t=30, b=95),
        plot_bgcolor="white",
    )
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits-dir", type=pathlib.Path, default=pathlib.Path("artifacts/fits"))
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("presentation/bandit/assets/model_comparison.html"))
    args = p.parse_args()

    fits = _load_fits(args.fits_dir)
    if not fits:
        raise SystemExit(f"no fits found in {args.fits_dir}")
    print(f"loaded {len(fits)} fits: {[f['model_name'] for f in fits]}")

    fig = build_accuracy_figure(fits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
