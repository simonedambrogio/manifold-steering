"""NHB Fig 1f reproduction: per-episode relative reward, partitioned into 5 task blocks.

Each block shown as a violin with the per-episode relative rewards as scatter and the
block mean as a black dot, with reference lines at 0 (random) and 1 (optimal-impossible).

    python -m figures.fig_aggregate_perf \
        --dataset artifacts/datasets/llama8b_dataset.json \
        --out artifacts/figures/fig1f.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from analyses.behavior import episode_relative_rewards, load_logs


def plot_fig1f(
    rr: np.ndarray, n_blocks: int = 5, model_name: str = "Llama-3.1-8B-Instruct"
) -> plt.Figure:
    chunks = np.array_split(rr, n_blocks)
    positions = np.arange(1, n_blocks + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    parts = ax.violinplot(
        [list(c) for c in chunks], positions=positions,
        showmeans=False, showmedians=False, showextrema=False, widths=0.85,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#cccccc"); body.set_edgecolor("#888888"); body.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for i, c in enumerate(chunks, 1):
        ax.scatter(
            i + rng.uniform(-0.18, 0.18, size=len(c)), c,
            s=12, alpha=0.45, color="#666666", linewidths=0,
        )
        ax.scatter(
            [i], [c.mean()], s=70, color="black", zorder=10,
            label="Group average" if i == 1 else None,
        )

    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.axhline(0.0, color="k", ls="--", lw=0.9)
    ax.annotate("Optimal choices\n(impossible)", xy=(n_blocks + 0.55, 1.0),
                xytext=(n_blocks + 0.55, 1.0), va="center", fontsize=9)
    ax.annotate("Random\nchoices", xy=(n_blocks + 0.55, 0.0),
                xytext=(n_blocks + 0.55, 0.0), va="center", fontsize=9)

    n_per = len(chunks[0])
    ax.set_xlabel("Episode block")
    ax.set_ylabel("Relative reward")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{(i-1)*n_per+1}–{i*n_per}" for i in positions])
    ax.set_xlim(0.4, n_blocks + 1.8)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title(f"Aggregate performance ({model_name}, {len(rr)} episodes)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    logs = load_logs(args.dataset)
    rr = episode_relative_rewards(logs)
    fig = plot_fig1f(rr, n_blocks=args.n_blocks, model_name=args.model_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
