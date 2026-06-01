"""Behavioral summaries of a collected LLM bandit dataset (numbers, no plotting).

Reports what's in a dataset file, exposes per-episode relative reward, and recovers the
ground-truth latent means μ[t, i] from the saved schedule seed (not stored directly).

    python -m analyses.behavior --dataset artifacts/datasets/llama8b_dataset.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from bandit.schedule import generate_schedule, relative_reward


def load_logs(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text())


def episode_relative_rewards(logs: list[dict]) -> np.ndarray:
    """Per-episode mean relative reward (n_episodes,)."""
    return np.array([
        relative_reward(np.array(l["actions"]), np.array(l["payoff_matrix"])).mean()
        for l in logs
    ])


def latent_means_for(log: dict) -> np.ndarray:
    """Ground-truth latent means μ[t, i], recovered from the saved schedule seed.

    Not stored in the log (redundant with payoff matrix + seed); the seed deterministically
    regenerates them via `bandit.schedule.generate_schedule`.
    """
    sched = generate_schedule(seed=log["schedule_seed"], n_trials=len(log["actions"]))
    return sched.means


def report_contents(logs: list[dict]) -> None:
    n = len(logs)
    T = len(logs[0]["actions"])
    n_arms = len(logs[0]["payoff_matrix"][0])
    print(f"{n} episodes × {T} trials")
    print(f"keys per episode: {list(logs[0].keys())}")
    print(f"  actions:        ({n}, {T})  int      chosen arm per trial")
    print(f"  rewards:        ({n}, {T})  int      reward delivered per trial")
    print(f"  choice_dists:   ({n}, {T}, {n_arms + 1})  float  model's P(arm) + 'other'")
    print(f"  payoff_matrix:  ({n}, {T}, {n_arms})  int     realized r[t,i] for all arms")
    print(f"  letters:        ({n}, {n_arms})  str        per-episode arm→label map")
    print(f"  schedule_seed:  ({n},)  int                    reward-schedule seed")
    print("\nground-truth latent means μ[t,i] are not stored directly; recover via")
    print("  `analyses.behavior.latent_means_for(log)`  (shape (T, n_arms))")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    args = parser.parse_args()

    logs = load_logs(args.dataset)
    print(f"loaded {args.dataset}")
    report_contents(logs)

    rr = episode_relative_rewards(logs)
    print(
        f"\nper-episode mean relative reward: {rr.mean():+.3f} ± {rr.std():.3f}  "
        f"(min {rr.min():+.2f}, max {rr.max():+.2f})"
    )


if __name__ == "__main__":
    main()
