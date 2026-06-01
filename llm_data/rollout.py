"""Closed-loop rollout harness: the LLM plays the restless 4-armed bandit.

Each trial: render the chat prompt from the raw (action -> outcome) history, read the
model's choice distribution after the `You press <<` prefill, sample an arm from that
distribution (the model's own policy), deliver the scheduled reward, append, repeat.
Produces per-episode logs (actions, rewards, per-trial choice distributions) for the
learning-gate check (plan Step 2) and the cognitive-model fit (Step 3).

Each episode uses a fresh reward schedule and a fresh randomized arm->letter map.

    python rollout.py --model Qwen/Qwen2.5-7B-Instruct --n-episodes 2 --n-trials 30

Note: this naively re-encodes the growing prompt each trial (O(T^2) tokens). Fine for
short runs; add KV-cache reuse before collecting many 150-trial episodes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass

import numpy as np

from bandit.schedule import BanditSchedule, generate_schedule, relative_reward
from llm_data import prompt as P
from llm_data.llm import (
    CachedEpisode,
    letter_token_ids,
    load_model,
    read_choice_distribution,
    read_choice_distributions_batched,
)


@dataclass
class EpisodeLog:
    actions: list[int]
    rewards: list[int]
    choice_dists: list[list[float]]  # [T, n_arms + 1]
    letters: list[str]
    payoff_matrix: list[list[int]]  # [T, n_arms] — for the relative-reward metric
    schedule_seed: int

    def relative_reward(self) -> np.ndarray:
        return relative_reward(np.array(self.actions), np.array(self.payoff_matrix))


def play_episode(
    model,
    tokenizer,
    schedule: BanditSchedule,
    label_map: P.LabelMap,
    rng: np.random.Generator,
    policy: str = "sample",
) -> EpisodeLog:
    """Play one episode; `policy` is "sample" (from the model's policy) or "greedy"."""
    label_ids = letter_token_ids(tokenizer, label_map.letters)
    n = label_map.n_arms
    history: list[tuple[int, int]] = []
    actions, rewards, dists = [], [], []
    for t in range(schedule.n_trials):
        prompt = P.build_chat_prompt(tokenizer, label_map, history)
        dist = read_choice_distribution(model, tokenizer, prompt, label_ids)
        p = dist[:n] / dist[:n].sum()
        a = int(rng.choice(n, p=p)) if policy == "sample" else int(np.argmax(p))
        r = int(schedule.reward(t, a))
        history.append((a, r))
        actions.append(a)
        rewards.append(r)
        dists.append([float(x) for x in dist])
    return EpisodeLog(
        actions=actions,
        rewards=rewards,
        choice_dists=dists,
        letters=label_map.letters,
        payoff_matrix=schedule.rewards.tolist(),
        schedule_seed=schedule.seed,
    )


def run_episodes(
    model,
    tokenizer,
    n_episodes: int,
    n_trials: int = 150,
    base_seed: int = 0,
    policy: str = "sample",
) -> list[EpisodeLog]:
    """Run several episodes, each with a fresh schedule and randomized labels."""
    logs = []
    for e in range(n_episodes):
        seed = base_seed + e
        schedule = generate_schedule(seed, n_trials=n_trials)
        rng = np.random.default_rng(10_000 + seed)  # labels + action sampling
        label_map = P.sample_label_map(rng, schedule.n_arms)
        logs.append(play_episode(model, tokenizer, schedule, label_map, rng, policy))
    return logs


def run_episodes_batched(
    model,
    tokenizer,
    n_episodes: int,
    n_trials: int = 150,
    base_seed: int = 0,
    policy: str = "sample",
) -> list[EpisodeLog]:
    """Run episodes in parallel: one batched forward pass per trial (all episodes step
    together). Much faster than sequential on parallel hardware. Each episode has its own
    schedule, randomized labels, and RNG; same return type as `run_episodes`.
    """
    schedules, label_maps, rngs, label_ids = [], [], [], []
    for e in range(n_episodes):
        seed = base_seed + e
        sched = generate_schedule(seed, n_trials=n_trials)
        rng = np.random.default_rng(10_000 + seed)
        lm = P.sample_label_map(rng, sched.n_arms)
        schedules.append(sched)
        rngs.append(rng)
        label_maps.append(lm)
        label_ids.append(letter_token_ids(tokenizer, lm.letters))

    n = schedules[0].n_arms
    histories = [[] for _ in range(n_episodes)]
    actions = [[] for _ in range(n_episodes)]
    rewards = [[] for _ in range(n_episodes)]
    dists = [[] for _ in range(n_episodes)]
    for t in range(n_trials):
        prompts = [
            P.build_chat_prompt(tokenizer, label_maps[e], histories[e])
            for e in range(n_episodes)
        ]
        D = read_choice_distributions_batched(model, tokenizer, prompts, label_ids)
        for e in range(n_episodes):
            p = D[e, :n] / D[e, :n].sum()
            a = int(rngs[e].choice(n, p=p)) if policy == "sample" else int(np.argmax(p))
            r = int(schedules[e].reward(t, a))
            histories[e].append((a, r))
            actions[e].append(a)
            rewards[e].append(r)
            dists[e].append([float(x) for x in D[e]])
    return [
        EpisodeLog(
            actions=actions[e],
            rewards=rewards[e],
            choice_dists=dists[e],
            letters=label_maps[e].letters,
            payoff_matrix=schedules[e].rewards.tolist(),
            schedule_seed=schedules[e].seed,
        )
        for e in range(n_episodes)
    ]


def play_episode_cached(
    model, tokenizer, schedule: BanditSchedule, label_map: P.LabelMap,
    rng: np.random.Generator, policy: str = "sample",
) -> EpisodeLog:
    """Play one episode using an incremental KV cache (fast). Same semantics as
    `play_episode`, but each trial only processes the new tokens."""
    ep = CachedEpisode(model, tokenizer, label_map)
    n = label_map.n_arms
    actions, rewards, dists = [], [], []
    for t in range(schedule.n_trials):
        dist = ep.choice_distribution()
        p = dist[:n] / dist[:n].sum()
        a = int(rng.choice(n, p=p)) if policy == "sample" else int(np.argmax(p))
        r = int(schedule.reward(t, a))
        ep.append(a, r)
        actions.append(a)
        rewards.append(r)
        dists.append([float(x) for x in dist])
    return EpisodeLog(
        actions=actions,
        rewards=rewards,
        choice_dists=dists,
        letters=label_map.letters,
        payoff_matrix=schedule.rewards.tolist(),
        schedule_seed=schedule.seed,
    )


def run_episodes_cached(
    model, tokenizer, n_episodes: int, n_trials: int = 150,
    base_seed: int = 0, policy: str = "sample",
) -> list[EpisodeLog]:
    """Sequential episodes, each KV-cached. Fast enough for large collection runs."""
    logs = []
    for e in range(n_episodes):
        seed = base_seed + e
        sched = generate_schedule(seed, n_trials=n_trials)
        rng = np.random.default_rng(10_000 + seed)
        lm = P.sample_label_map(rng, sched.n_arms)
        logs.append(play_episode_cached(model, tokenizer, sched, lm, rng, policy))
    return logs


def learning_curve(logs: list[EpisodeLog]) -> np.ndarray:
    """Mean per-trial relative reward across episodes (episodes must share n_trials)."""
    return np.stack([log.relative_reward() for log in logs]).mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run closed-loop LLM bandit episodes and report the learning curve.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--policy", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--batched", action="store_true", help="batch episodes per trial")
    parser.add_argument("--cached", action="store_true", help="KV-cached per-episode (fast)")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="save logs as JSON")
    args = parser.parse_args()

    model, tok = load_model(args.model)
    print(f"loaded {args.model} on {model.device}")
    runner = (
        run_episodes_cached if args.cached
        else run_episodes_batched if args.batched
        else run_episodes
    )
    logs = runner(model, tok, args.n_episodes, args.n_trials, args.base_seed, args.policy)

    for i, log in enumerate(logs):
        rr = log.relative_reward()
        print(
            f"episode {i}: mean relative reward = {rr.mean():+.3f}  "
            f"(first third {rr[: len(rr) // 3].mean():+.3f} -> "
            f"last third {rr[-len(rr) // 3:].mean():+.3f})"
        )
    curve = learning_curve(logs)
    print("\nlearning curve (relative reward, binned in fifths):")
    bins = np.array_split(curve, 5)
    print("  " + "  ".join(f"{b.mean():+.2f}" for b in bins))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(log) for log in logs]))
        print(f"\nsaved {len(logs)} episode logs -> {args.out}")


if __name__ == "__main__":
    main()
