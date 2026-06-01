"""Visualize per-trial Q trajectories from a fitted cognitive model.

Plots the latent Q for a few example episodes (one line per arm) with the chosen action
strip and the schedule's true latent means overlaid as dashed lines. Useful as a sanity
check that the recovered Q tracks observed rewards and that argmax(Q) approximately
matches Llama's choices.

    python -m figures.fig_value_traces \
        --q artifacts/value/bestrl_Q.pt \
        --out artifacts/figures/fig_value_traces.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from bandit.schedule import generate_schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--q", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="episode indices to plot")
    args = parser.parse_args()

    data = torch.load(args.q, weights_only=False)
    Q = data["Q"]                          # [n_eps, T, n_arms]
    actions = data["actions"]              # [n_eps, T]
    seeds = data["schedule_seeds"]
    model_name = data["model_name"]
    n_arms = Q.shape[2]
    T = Q.shape[1]
    colors = plt.cm.tab10(np.arange(n_arms))

    n_rows = len(args.episodes)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(13, 2.2 * n_rows), sharex=True, squeeze=False,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    for row, ep in enumerate(args.episodes):
        q_ep = Q[ep].numpy()
        act_ep = actions[ep].numpy()
        sched = generate_schedule(seed=seeds[ep], n_trials=T)
        mu = sched.means                   # [T, n_arms]

        ax_q, ax_mu = axes[row]
        for i in range(n_arms):
            ax_q.plot(q_ep[:, i], color=colors[i], lw=1.4, label=f"arm {i}")
            ax_mu.plot(mu[:, i], color=colors[i], lw=1.4)

        # chosen-action strip beneath each panel (small colored markers)
        y_strip_q = q_ep.min() - 0.04 * (q_ep.max() - q_ep.min() + 1)
        y_strip_mu = mu.min() - 0.04 * (mu.max() - mu.min() + 1)
        ax_q.scatter(np.arange(T), [y_strip_q] * T, c=[colors[a] for a in act_ep],
                     s=6, marker="s", linewidths=0)
        ax_mu.scatter(np.arange(T), [y_strip_mu] * T, c=[colors[a] for a in act_ep],
                      s=6, marker="s", linewidths=0)

        ax_q.set_ylabel("BestRL Q")
        ax_mu.set_ylabel("μ (schedule)")
        ax_q.set_title(f"episode {ep}", loc="left", fontsize=9)
        if row == 0:
            ax_q.legend(loc="upper right", fontsize=8, ncol=4)

    axes[-1, 0].set_xlabel("trial")
    axes[-1, 1].set_xlabel("trial")
    fig.suptitle(
        f"{model_name} recovered Q (left) vs schedule's true latent means (right)\n"
        "colored strip = arm chosen on each trial",
        fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
