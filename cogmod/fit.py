"""Generic fitter for `BehavioralModel` subclasses.

**Loss = distribution cross-entropy** (`dist_ce`): the cognitive model's predicted
choice distribution is matched against the LLM's saved per-trial distribution
(`dist_targets`). This is a strict improvement over action-NLL fitting whenever the
distribution is available, since it eliminates the sampling-noise variance.

- Small RL models (`SimpleRL`, `BestRL`): **multi-start `torch.optim.LBFGS`**.
- RNN models (`VanillaRNN`, `MemoryANN`): **`torch.optim.AdamW` + early stopping on val**
  + a small (seed × hyperparameter) sweep.

`FitResult` also reports the action-based NLL/accuracy for NHB-style comparison.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cogmod.base import EPS, BehavioralModel, Dataset, FitResult, bic_score


def _n_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _evaluate(model: BehavioralModel, ds: Dataset) -> dict:
    """Compute all reporting metrics for one split."""
    model.eval()
    ce = float(model.dist_ce(ds.actions, ds.rewards, ds.dist_targets))
    h_target = float(BehavioralModel.target_entropy(ds.dist_targets))
    kl = ce - h_target
    nll = float(model.nll(ds.actions, ds.rewards))
    acc = math.exp(-nll / ds.n_predict_trials)
    return {"ce": ce, "kl": kl, "nll": nll, "acc": acc}


@torch.no_grad()
def _per_episode_kl(model: BehavioralModel, ds: Dataset) -> list[float]:
    """Per-episode mean-per-trial KL (lower = better; 0 = perfect)."""
    model.eval()
    probs = model.forward(ds.actions, ds.rewards)
    log_pred = torch.log(probs[:, 1:].clamp(min=EPS))
    log_tgt = torch.log(ds.dist_targets[:, 1:].clamp(min=EPS))
    ce = -(ds.dist_targets[:, 1:] * log_pred).sum(dim=-1)   # [B, T-1]
    h = -(ds.dist_targets[:, 1:] * log_tgt).sum(dim=-1)     # [B, T-1]
    return (ce - h).mean(dim=1).tolist()


@torch.no_grad()
def _per_episode_accuracy(model: BehavioralModel, ds: Dataset) -> list[float]:
    """Per-episode action accuracy `exp(-NLL_per_episode / n_predict_trials)`."""
    model.eval()
    probs = model.forward(ds.actions, ds.rewards)
    a_onehot = F.one_hot(ds.actions, model.n_arms).float()
    chosen_p = (probs[:, 1:] * a_onehot[:, 1:]).sum(dim=-1).clamp(min=EPS)
    nll = -torch.log(chosen_p).sum(dim=1)
    return torch.exp(-nll / ds.n_predict_trials).tolist()


def _build_result(
    model: BehavioralModel, model_name: str,
    train: Dataset, val: Dataset, test: Dataset,
    metrics: dict[str, dict],   # {"train": {...}, "val": {...}, "test": {...}}
    extras: dict,
) -> FitResult:
    n_params = _n_trainable_params(model)
    return FitResult(
        model_name=model_name,
        n_params=n_params,
        n_train_episodes=train.n_episodes,
        train_ce=metrics["train"]["ce"], val_ce=metrics["val"]["ce"], test_ce=metrics["test"]["ce"],
        train_kl=metrics["train"]["kl"], val_kl=metrics["val"]["kl"], test_kl=metrics["test"]["kl"],
        train_nll=metrics["train"]["nll"], val_nll=metrics["val"]["nll"], test_nll=metrics["test"]["nll"],
        train_acc=metrics["train"]["acc"], val_acc=metrics["val"]["acc"], test_acc=metrics["test"]["acc"],
        bic=bic_score(metrics["train"]["ce"], n_params, train.n_episodes, train.n_predict_trials),
        params=model.params_dict(),
        test_per_episode_kl=_per_episode_kl(model, test),
        test_per_episode_acc=_per_episode_accuracy(model, test),
        extras=extras,
    )


# --- RL fitter ---------------------------------------------------------------


def fit_rl_multistart(
    model_cls: type[BehavioralModel],
    train: Dataset, val: Dataset, test: Dataset,
    n_starts: int = 30,
    max_iter: int = 100,
    seed: int = 0,
    init_jitter: float = 0.5,
    verbose: bool = False,
) -> FitResult:
    """Multi-start L-BFGS for small RL models. Selects by validation CE (= KL up to constant)."""
    rng = np.random.default_rng(seed)
    best: dict | None = None
    n_failed = 0

    for start in range(n_starts):
        torch.manual_seed(int(rng.integers(0, 1_000_000)))
        model = model_cls()
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * init_jitter)

        optim = torch.optim.LBFGS(
            model.parameters(),
            max_iter=max_iter, line_search_fn="strong_wolfe",
            tolerance_grad=1e-8, tolerance_change=1e-10,
        )

        def closure():
            optim.zero_grad()
            loss = model.dist_ce(train.actions, train.rewards, train.dist_targets)
            loss.backward()
            return loss

        try:
            optim.step(closure)
        except (RuntimeError, ValueError):
            n_failed += 1
            continue

        val_m = _evaluate(model, val)
        if verbose:
            print(f"  start {start}: val CE {val_m['ce']:.2f} KL {val_m['kl']:.2f}  params {model.params_dict()}")

        if best is None or val_m["ce"] < best["val"]["ce"]:
            train_m = _evaluate(model, train)
            test_m = _evaluate(model, test)
            best = {"model": model, "train": train_m, "val": val_m, "test": test_m}

    if best is None:
        raise RuntimeError(f"All {n_starts} L-BFGS starts failed for {model_cls.__name__}")

    extras = {
        "n_starts": n_starts, "n_failed": n_failed,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in best["model"].state_dict().items()},
    }
    return _build_result(
        best["model"], model_cls.__name__, train, val, test,
        {"train": best["train"], "val": best["val"], "test": best["test"]},
        extras,
    )


# --- RNN fitter --------------------------------------------------------------


_DEFAULT_RNN_HP_GRID = [
    {"hidden_size": 16, "lr": 1e-4},
    {"hidden_size": 16, "lr": 5e-4},
    {"hidden_size": 32, "lr": 1e-4},
    {"hidden_size": 32, "lr": 5e-4},
]


def fit_rnn(
    model_cls: type[BehavioralModel],
    train: Dataset, val: Dataset, test: Dataset,
    hp_grid: list[dict] | None = None,
    n_seeds: int = 5,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 20,
    weight_decay: float = 1e-5,
    seed: int = 0,
    verbose: bool = False,
) -> FitResult:
    """AdamW + early stopping on val + (seed × HP) sweep. Best (combo, seed) by val CE."""
    grid = hp_grid if hp_grid is not None else _DEFAULT_RNN_HP_GRID
    rng = np.random.default_rng(seed)
    best: dict | None = None
    n_fits = 0

    for hp in grid:
        hidden_size = hp["hidden_size"]
        lr = hp["lr"]
        for seed_i in range(n_seeds):
            torch.manual_seed(int(rng.integers(0, 1_000_000)))
            model = model_cls(hidden_size=hidden_size)
            optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_ce = float("inf")
            best_state = None
            no_improve = 0
            last_epoch = 0
            for epoch in range(max_epochs):
                last_epoch = epoch
                model.train()
                perm = torch.randperm(train.n_episodes)
                for i in range(0, train.n_episodes, batch_size):
                    idx = perm[i : i + batch_size]
                    optim.zero_grad()
                    loss = model.dist_ce(
                        train.actions[idx], train.rewards[idx], train.dist_targets[idx],
                    )
                    loss.backward()
                    optim.step()

                val_m_i = _evaluate(model, val)
                if val_m_i["ce"] < best_val_ce - 1e-4:
                    best_val_ce = val_m_i["ce"]
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            if best_state is None:
                continue
            model.load_state_dict(best_state)
            train_m = _evaluate(model, train)
            val_m = _evaluate(model, val)
            n_fits += 1

            if verbose:
                print(
                    f"  hp={hp} seed={seed_i}: val_CE={val_m['ce']:.2f} KL={val_m['kl']:.2f}  "
                    f"(stopped at epoch {last_epoch + 1})"
                )

            if best is None or val_m["ce"] < best["val"]["ce"]:
                test_m = _evaluate(model, test)
                best = {
                    "model": model, "hp": hp, "seed": seed_i, "epochs": last_epoch + 1,
                    "train": train_m, "val": val_m, "test": test_m,
                }

    if best is None:
        raise RuntimeError(f"All {n_fits} RNN fits failed for {model_cls.__name__}")

    extras = {
        "best_hp": best["hp"], "best_seed": best["seed"], "best_epochs": best["epochs"],
        "n_fits": n_fits,
    }
    return _build_result(
        best["model"], model_cls.__name__, train, val, test,
        {"train": best["train"], "val": best["val"], "test": best["test"]},
        extras,
    )


# --- dispatch ---------------------------------------------------------------


def fit_model(
    model_cls: type[BehavioralModel], kind: str,
    train: Dataset, val: Dataset, test: Dataset,
    **kwargs,
) -> FitResult:
    """Dispatch to the right fitter. `kind` ∈ {'rl', 'rnn'}."""
    if kind == "rl":
        return fit_rl_multistart(model_cls, train, val, test, **kwargs)
    if kind == "rnn":
        return fit_rnn(model_cls, train, val, test, **kwargs)
    raise ValueError(f"unknown fit kind: {kind!r}")
