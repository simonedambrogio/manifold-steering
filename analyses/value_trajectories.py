"""Extract per-trial Q trajectories from a fitted cognitive model.

Loads a `FitResult` JSON (with `extras.state_dict`), reconstitutes the model, runs
`value_trajectory()` on every episode in the dataset, and saves the resulting
`[n_episodes, T, n_arms]` tensor of latent Q values — the quantity we'll probe in the
LLM's activations during Step 4 (locate).

    python -m analyses.value_trajectories \
        --fit artifacts/fits/bestrl.json \
        --dataset artifacts/datasets/llama8b_dataset.json \
        --out artifacts/value/bestrl_Q.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from cogmod.base import BehavioralModel
from cogmod.best_rl import BestRL
from cogmod.memory_ann import MemoryANN
from cogmod.simple_rl import SimpleRL
from cogmod.vanilla_rnn import VanillaRNN


MODEL_REGISTRY: dict[str, type[BehavioralModel]] = {
    "SimpleRL":   SimpleRL,
    "BestRL":     BestRL,
    "VanillaRNN": VanillaRNN,
    "MemoryANN":  MemoryANN,
}


def _restore_model(fit: dict) -> BehavioralModel:
    """Rebuild a model from a saved FitResult JSON (uses `extras.state_dict`)."""
    name = fit["model_name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model: {name}; registry has {list(MODEL_REGISTRY)}")

    sd_raw = fit.get("extras", {}).get("state_dict")
    if sd_raw is None:
        raise KeyError(
            f"{name}: no state_dict in extras — RL fits include it; RNN fits don't "
            "(too large for JSON). Re-fit and save state_dict separately for RNN models."
        )

    # RNN models need hidden_size from the fitted hyperparams to instantiate the right shape.
    init_kwargs: dict = {}
    if name in ("VanillaRNN", "MemoryANN"):
        hp = fit.get("extras", {}).get("best_hp", {})
        if "hidden_size" in hp:
            init_kwargs["hidden_size"] = hp["hidden_size"]

    model = MODEL_REGISTRY[name](**init_kwargs)
    state_dict = {k: torch.tensor(v) for k, v in sd_raw.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_all_episodes(dataset_path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Load every episode (not split) as (actions, rewards, schedule_seeds)."""
    logs = json.loads(dataset_path.read_text())
    actions = torch.tensor([l["actions"] for l in logs], dtype=torch.long)
    rewards = torch.tensor([l["rewards"] for l in logs], dtype=torch.float32)
    seeds = [l["schedule_seed"] for l in logs]
    return actions, rewards, seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fit", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fit = json.loads(args.fit.read_text())
    model = _restore_model(fit)
    actions, rewards, seeds = _load_all_episodes(args.dataset)

    with torch.no_grad():
        Q = model.value_trajectory(actions, rewards)  # [n_eps, T, n_arms]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "Q":           Q.cpu(),               # [n_eps, T, n_arms]: value BEFORE choice at each trial
        "actions":     actions,               # [n_eps, T] for downstream alignment
        "rewards":     rewards,               # [n_eps, T]
        "schedule_seeds": seeds,              # list[int], one per episode
        "model_name":  fit["model_name"],
        "model_params": fit["params"],
        "n_episodes":  Q.shape[0],
        "n_trials":    Q.shape[1],
        "n_arms":      Q.shape[2],
    }, args.out)

    print(f"reconstituted {fit['model_name']} with params: {fit['params']}")
    print(f"saved Q shape {tuple(Q.shape)}  range [{float(Q.min()):.1f}, {float(Q.max()):.1f}]  → {args.out}")


if __name__ == "__main__":
    main()
