"""Fit all four cognitive models to an LLM bandit dataset and compare test performance.

Fits the RL pair (`SimpleRL`, `BestRL`) with multi-start L-BFGS and the RNN pair
(`VanillaRNN`, `MemoryANN`) with AdamW + early stopping + a small seed × HP sweep.
Saves a `FitResult` JSON per model (with `test_per_episode_acc` for the Fig 2d plot)
into `--out-dir`.

    python -m analyses.model_compare \
        --dataset artifacts/datasets/llama8b_dataset.json \
        --out-dir artifacts/fits/
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

import torch

from cogmod.base import FitResult, load_dataset
from cogmod.best_rl import BestRL
from cogmod.fit import fit_model
from cogmod.memory_ann import MemoryANN
from cogmod.simple_rl import SimpleRL
from cogmod.vanilla_rnn import VanillaRNN


MODELS = [
    # (class,       kind,  per-fit kwargs)
    (SimpleRL,    "rl",  {"n_starts": 30}),
    (BestRL,      "rl",  {"n_starts": 50}),
    (VanillaRNN,  "rnn", {}),
    (MemoryANN,   "rnn", {}),
]


def _save(res: FitResult, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(res), f, indent=2)


def _print_table(results: list[FitResult]) -> None:
    print(
        f"\n{'model':<14}{'k':>5}{'train_CE':>11}{'val_CE':>9}"
        f"{'test_CE':>9}{'test_KL':>9}{'test_acc':>10}{'BIC':>12}"
    )
    print("-" * 79)
    for r in results:
        print(
            f"{r.model_name:<14}{r.n_params:>5}{r.train_ce:>11.1f}"
            f"{r.val_ce:>9.1f}{r.test_ce:>9.1f}{r.test_kl:>9.2f}"
            f"{r.test_acc:>10.3f}{r.bic:>12.1f}"
        )
    print(
        "\nNote: CE is the fit loss (lower=better; ceiling = H(target) ≈ 0.67 nats·149 trials ≈ 99). "
        "KL = CE − H(target) (0=perfect). test_acc is the action-based NHB-style metric (kept for comparison)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                        help="directory to save per-model FitResult JSONs")
    parser.add_argument("--only", nargs="+", default=None,
                        help="limit to a subset of models (by class name)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    train, val, test = load_dataset(args.dataset)
    print(f"dataset: {args.dataset}")
    print(
        f"  train {train.n_episodes}×{train.n_trials}  "
        f"val {val.n_episodes}×{val.n_trials}  "
        f"test {test.n_episodes}×{test.n_trials}  "
        f"(predicting trials 1..{train.n_trials - 1})"
    )

    selected = MODELS if args.only is None else [
        m for m in MODELS if m[0].__name__ in set(args.only)
    ]

    results: list[FitResult] = []
    for model_cls, kind, kwargs in selected:
        tag = "multi-start L-BFGS" if kind == "rl" else "AdamW + early-stop + HP sweep"
        print(f"\n=== Fitting {model_cls.__name__} ({tag}) ===")
        res = fit_model(model_cls, kind, train, val, test, verbose=args.verbose, **kwargs)
        print(res)
        print(f"  params: {res.params}")
        results.append(res)
        if args.out_dir is not None:
            _save(res, args.out_dir / f"{model_cls.__name__.lower()}.json")

    _print_table(results)


if __name__ == "__main__":
    main()
