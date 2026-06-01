"""Collect a dataset of LLM bandit episodes (KV-cached), saving incrementally.

Each episode uses a fresh reward schedule and randomized labels; the LLM plays closed-loop
and we log actions, rewards, and per-trial choice distributions. Saves the running list to
`--out` every `--save-every` episodes so a long run is crash-safe. This is the canonical
dataset for cognitive-model fitting (Step 3) and the behavior manifold later.

    python collect.py --model meta-llama/Llama-3.1-8B-Instruct \
        --n-episodes 400 --n-trials 150 --out runs/llama8b_dataset.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import asdict

import numpy as np

from bandit.schedule import generate_schedule
from llm_data import prompt as P
from llm_data.llm import load_model
from llm_data.rollout import play_episode_cached


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect LLM bandit episodes (KV-cached, incremental save).",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-episodes", type=int, default=400)
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--fixed-letters", nargs="+", default=None,
                        help="freeze the arm→letter map across episodes (e.g. B C D F); "
                             "reward schedules still vary per episode")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    model, tok = load_model(args.model)
    print(f"loaded {args.model} on {model.device}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    t0 = time.time()
    for e in range(args.n_episodes):
        seed = args.base_seed + e
        sched = generate_schedule(seed, n_trials=args.n_trials)
        rng = np.random.default_rng(10_000 + seed)
        lm = (P.LabelMap(letters=list(args.fixed_letters)) if args.fixed_letters
              else P.sample_label_map(rng, sched.n_arms))
        logs.append(play_episode_cached(model, tok, sched, lm, rng))

        if (e + 1) % args.save_every == 0 or e == args.n_episodes - 1:
            args.out.write_text(json.dumps([asdict(x) for x in logs]))
            rr = float(np.mean([x.relative_reward().mean() for x in logs]))
            elapsed = time.time() - t0
            eta = elapsed / (e + 1) * (args.n_episodes - e - 1)
            print(
                f"[{e + 1}/{args.n_episodes}] saved; mean rel reward {rr:+.3f}; "
                f"elapsed {elapsed / 60:.1f} min; eta {eta / 60:.1f} min",
                flush=True,
            )

    print(f"done: {len(logs)} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
