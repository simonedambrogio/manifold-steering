"""Model comparison figure (NHB Fig 2d-style, two panels).

Reads `FitResult` JSONs from `--fits-dir` and produces a 1×2 figure:

  Left  — **Mean per-trial KL** (lower = better; 0 = perfect distribution match).
          This is the primary metric (matches the new dist-CE fit objective).
  Right — **Action accuracy %** (the NHB Fig 2d metric, kept for comparison).

Both panels: one point per model with error bars = SEM across the 40 test episodes;
significance brackets between adjacent models from paired t-tests.

    python -m figures.fig_model_comparison \
        --fits-dir artifacts/fits/ \
        --out artifacts/figures/fig_model_comparison.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Display order — cognitive ladder, simplest → most flexible
MODEL_ORDER = ["SimpleRL", "BestRL", "VanillaRNN", "MemoryANN"]


def _load_fits(fits_dir: pathlib.Path) -> list[dict]:
    fits = {}
    for p in fits_dir.glob("*.json"):
        d = json.loads(p.read_text())
        fits[d["model_name"]] = d
    return [fits[name] for name in MODEL_ORDER if name in fits]


def _p_label(p: float) -> str:
    if p < 0.001:
        return "P < 0.001"
    if p < 0.05:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"


def _panel(
    ax: plt.Axes,
    names: list[str],
    per_eps_data: list[list[float] | None],
    aggregate_fallback: list[float],
    ylabel: str,
    title: str,
    lower_is_better: bool,
    ref_line: tuple[float, str] | None = None,
) -> None:
    """One subpanel: dot per model + error bar + brackets."""
    means, sems, per_eps_arr = [], [], []
    for i, raw in enumerate(per_eps_data):
        if raw:
            a = np.asarray(raw, dtype=float)
            means.append(float(a.mean()))
            sems.append(float(a.std(ddof=1) / np.sqrt(len(a))))
            per_eps_arr.append(a)
        else:
            means.append(float(aggregate_fallback[i]))
            sems.append(0.0)
            per_eps_arr.append(None)
            print(f"  ⚠ {names[i]}: no per-episode data for {ylabel} (mean only)")

    x = np.arange(len(names))
    ax.errorbar(x, means, yerr=sems, fmt="o", color="black", capsize=4, markersize=8, lw=1.2)

    # Significance brackets between adjacent models
    y_top = max(m + s for m, s in zip(means, sems))
    y_bot = min(m - s for m, s in zip(means, sems))
    span = max(y_top - y_bot, 1e-3)
    bracket_h = 0.06 * span
    y = y_top + 0.04 * span
    for i in range(len(names) - 1):
        if per_eps_arr[i] is None or per_eps_arr[i + 1] is None:
            continue
        # Always test "better" direction: for lower-is-better metrics test if i has lower mean than i+1
        t, p = stats.ttest_rel(per_eps_arr[i], per_eps_arr[i + 1])
        ax.plot([i, i, i + 1, i + 1], [y, y + bracket_h, y + bracket_h, y], "k-", lw=0.9)
        ax.text((2 * i + 1) / 2, y + bracket_h + 0.02 * span, _p_label(float(p)),
                ha="center", fontsize=8)
        y += bracket_h + 0.10 * span

    if ref_line is not None:
        val, label = ref_line
        ax.axhline(val, color="gray", ls="--", lw=0.7)
        ax.text(len(names) - 0.5, val, "  " + label, color="gray", fontsize=8,
                ha="right", va="bottom" if lower_is_better else "top")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(min(y_bot - 0.2 * span, y_bot), y + 0.1 * span)


def plot_fig_model_comparison(fits: list[dict]) -> plt.Figure:
    names = [f["model_name"] for f in fits]

    fig, (ax_kl, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Left panel: KL (primary metric, lower = better, 0 = perfect)
    _panel(
        ax_kl, names,
        per_eps_data=[f.get("test_per_episode_kl") for f in fits],
        aggregate_fallback=[f.get("test_kl", 0.0) for f in fits],
        ylabel="Mean per-trial KL (nats; lower = better)",
        title="Distribution fit quality",
        lower_is_better=True,
        ref_line=(0.0, "perfect"),
    )

    # Right panel: action accuracy (NHB Fig 2d metric, kept for comparison)
    _panel(
        ax_acc, names,
        per_eps_data=[f.get("test_per_episode_acc") for f in fits],
        aggregate_fallback=[f.get("test_acc", 0.25) for f in fits],
        ylabel="Action accuracy",
        title="Action prediction (NHB Fig 2d style)",
        lower_is_better=False,
        ref_line=(0.25, "chance"),
    )

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fits-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fits = _load_fits(args.fits_dir)
    if not fits:
        raise SystemExit(f"no fits found in {args.fits_dir}")
    print(f"loaded {len(fits)} fits: {[f['model_name'] for f in fits]}")

    fig = plot_fig_model_comparison(fits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
