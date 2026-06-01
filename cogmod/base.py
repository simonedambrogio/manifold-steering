"""Common interface for cognitive / neural behavioral models on the bandit.

A `BehavioralModel` maps a sequence of `(action, reward)` inputs to a sequence of
next-action probabilities. Subclasses implement just three methods:

  initial_state(B) -> state
  policy(state)    -> action_probs [B, n_arms]
  update(state, action_onehot, reward) -> new_state

The base class then provides `forward`, the two loss variants `nll` (action-based,
behavioral cloning — kept for NHB-style comparison) and `dist_ce` (distribution-based,
the cognitive model is fit to the LLM's full per-trial choice distribution — strict
improvement when the distribution is available), and `accuracy` / `target_entropy`
helpers.

**Indexing convention.** `probs[t] = P(action_t | history before t)`. Losses are summed
over predicted trials 1..T-1; trial 0's prediction comes from the prior alone and is
excluded, matching hybrid-rnn's training loss.
"""

from __future__ import annotations

import json
import math
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-12


# ---- Dataset loading -------------------------------------------------------


@dataclass
class Dataset:
    """A train/val/test split of bandit episodes."""

    actions: torch.Tensor      # [B, T] int64 — chosen arm 0..n_arms-1
    rewards: torch.Tensor      # [B, T] float32
    dist_targets: torch.Tensor # [B, T, n_arms] float32 — LLM's per-trial 4-arm policy
                               #   (choice_dists with the tiny 'other' mass renormalized out)
    n_arms: int
    name: str = ""

    @property
    def n_episodes(self) -> int:
        return self.actions.shape[0]

    @property
    def n_trials(self) -> int:
        return self.actions.shape[1]

    @property
    def n_predict_trials(self) -> int:
        """Trials we score the NLL / CE on (excludes trial 0; see indexing convention)."""
        return self.n_trials - 1


def load_dataset(
    path: pathlib.Path,
    device: str = "cpu",
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[Dataset, Dataset, Dataset]:
    """Load a collected dataset (JSON) → train/val/test by-episode 80/10/10.

    Splits are deterministic and contiguous. Also extracts `dist_targets` from the saved
    `choice_dists`, dropping the tiny `'other'` mass and renormalizing — this gives the
    LLM's per-trial 4-arm policy that the cognitive models are fit against.
    """
    logs = json.loads(pathlib.Path(path).read_text())
    n_arms = len(logs[0]["payoff_matrix"][0])
    actions = torch.tensor([l["actions"] for l in logs], dtype=torch.long, device=device)
    rewards = torch.tensor([l["rewards"] for l in logs], dtype=torch.float32, device=device)

    # choice_dists is [B, T, n_arms + 1]; renormalize the n_arms-vector to drop 'other'.
    choice_dists = torch.tensor(
        [l["choice_dists"] for l in logs], dtype=torch.float32, device=device
    )
    dist_targets = choice_dists[..., :n_arms]
    dist_targets = dist_targets / dist_targets.sum(dim=-1, keepdim=True).clamp(min=EPS)

    n = actions.shape[0]
    n_train = int(round(split[0] * n))
    n_val = int(round(split[1] * n))
    sl = lambda lo, hi: slice(lo, hi)
    return (
        Dataset(
            actions[sl(0, n_train)], rewards[sl(0, n_train)], dist_targets[sl(0, n_train)],
            n_arms, "train",
        ),
        Dataset(
            actions[sl(n_train, n_train + n_val)],
            rewards[sl(n_train, n_train + n_val)],
            dist_targets[sl(n_train, n_train + n_val)],
            n_arms, "val",
        ),
        Dataset(
            actions[sl(n_train + n_val, n)],
            rewards[sl(n_train + n_val, n)],
            dist_targets[sl(n_train + n_val, n)],
            n_arms, "test",
        ),
    )


# ---- Bounded-param reparameterizations (mirror hybrid-rnn) ------------------


def unit_param(value: float = 0.5) -> nn.Parameter:
    """Free parameter in (0, 1) via `sigmoid` (matches hybrid-rnn `unsigmoid_*`)."""
    logit = float(np.log(value / (1 - value))) if 0 < value < 1 else 0.0
    return nn.Parameter(torch.tensor(logit, dtype=torch.float32))


def positive_param(value: float = 1.0) -> nn.Parameter:
    """Free parameter in (0, ∞) via `softplus` (cleaner than relu used in hybrid-rnn)."""
    raw = float(np.log(np.expm1(value))) if value > 0 else 0.0
    return nn.Parameter(torch.tensor(raw, dtype=torch.float32))


def signed_unit_param(value: float = 0.0) -> nn.Parameter:
    """Free parameter in (-1, 1) via `tanh` (matches hybrid-rnn `untanh_*`)."""
    raw = float(np.arctanh(np.clip(value, -0.99, 0.99)))
    return nn.Parameter(torch.tensor(raw, dtype=torch.float32))


def real_param(value: float = 0.0) -> nn.Parameter:
    """Unbounded scalar parameter."""
    return nn.Parameter(torch.tensor(value, dtype=torch.float32))


# ---- The ABC ----------------------------------------------------------------


class BehavioralModel(nn.Module, ABC):
    """Subclass interface: `initial_state`, `policy`, `update`. Base handles the rest."""

    n_arms: int

    @abstractmethod
    def initial_state(self, batch_size: int):
        ...

    @abstractmethod
    def policy(self, state) -> torch.Tensor:
        """`state -> action_probs [B, n_arms]`."""

    @abstractmethod
    def update(self, state, action_onehot: torch.Tensor, reward: torch.Tensor):
        """Apply trial-`t` input to state; return new state. Called between `policy`s."""

    def forward(self, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Run the full sequence; return `action_probs [B, T, n_arms]`.

        `action_probs[t]` is computed from the state going *into* trial `t` (before
        seeing `(action_t, reward_t)`), so it represents `P(action_t | history < t)`.
        """
        B, T = actions.shape
        a_onehot = F.one_hot(actions, self.n_arms).float()
        state = self.initial_state(B)
        outs = []
        for t in range(T):
            outs.append(self.policy(state))
            state = self.update(state, a_onehot[:, t], rewards[:, t])
        return torch.stack(outs, dim=1)

    # ----- losses -----

    def nll(self, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Behavioral-cloning NLL: `-Σ_{t=1..T-1} log P(a_t)`, mean over batch."""
        probs = self.forward(actions, rewards)
        a_onehot = F.one_hot(actions, self.n_arms).float()
        chosen_p = (probs[:, 1:] * a_onehot[:, 1:]).sum(dim=-1).clamp(min=EPS)
        return -torch.log(chosen_p).sum(dim=1).mean()

    def dist_ce(
        self, actions: torch.Tensor, rewards: torch.Tensor, dist_targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy between predicted choice distribution and the LLM's actual
        per-trial distribution `dist_targets`. Sum over predicted trials, mean over batch.

        Mathematically `CE = KL(target || pred) + H(target)`; minimizing CE = minimizing
        KL, with much lower estimator variance than the action NLL (the latter is an
        unbiased but high-variance estimator of the same gradient).
        """
        probs = self.forward(actions, rewards)
        log_pred = torch.log(probs[:, 1:].clamp(min=EPS))
        ce = -(dist_targets[:, 1:] * log_pred).sum(dim=-1)  # [B, T-1]
        return ce.sum(dim=1).mean()

    # ----- diagnostics -----

    @staticmethod
    def target_entropy(dist_targets: torch.Tensor) -> torch.Tensor:
        """`H(target)` per trial summed over predicted trials, mean over batch.

        The CE achieves this value when the prediction equals the target — so
        `KL = CE − H(target)`.
        """
        log_t = torch.log(dist_targets[:, 1:].clamp(min=EPS))
        h = -(dist_targets[:, 1:] * log_t).sum(dim=-1)
        return h.sum(dim=1).mean()

    @torch.no_grad()
    def accuracy(self, actions: torch.Tensor, rewards: torch.Tensor) -> float:
        """`exp(-NLL / n_predict_trials)` — NHB Fig 2d predictive accuracy metric (action-based)."""
        nll = float(self.nll(actions, rewards))
        return float(math.exp(-nll / (actions.shape[1] - 1)))

    @torch.no_grad()
    def value_trajectory(self, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Per-trial latent value `[B, T, n_arms]` driving each choice.

        Default: re-runs the loop and stacks `_value_of(state)` at each `t`. Subclasses
        whose state isn't directly the value tensor should override `_value_of`.
        """
        B, T = actions.shape
        a_onehot = F.one_hot(actions, self.n_arms).float()
        state = self.initial_state(B)
        traj = []
        for t in range(T):
            traj.append(self._value_of(state).clone())
            state = self.update(state, a_onehot[:, t], rewards[:, t])
        return torch.stack(traj, dim=1)

    def _value_of(self, state) -> torch.Tensor:
        """Override if `state` is not directly the value tensor."""
        return state

    def params_dict(self) -> dict[str, float]:
        """Return the named scalar parameter values (override per subclass)."""
        return {n: float(p) for n, p in self.named_parameters() if p.numel() == 1}


# ---- FitResult --------------------------------------------------------------


@dataclass
class FitResult:
    """Result of fitting one behavioral model. Reports both the new distribution-based
    metrics (the fit objective) and the older action-based metrics (NHB-style)."""

    model_name: str
    n_params: int
    n_train_episodes: int

    # Distribution-based — the fit objective
    train_ce: float; val_ce: float; test_ce: float           # cross-entropy (lower = better)
    train_kl: float; val_kl: float; test_kl: float           # KL = CE − H(target); 0 = perfect

    # Action-based — kept for NHB-style comparison
    train_nll: float; val_nll: float; test_nll: float
    train_acc: float; val_acc: float; test_acc: float

    bic: float                                                # uses the fit loss (CE)
    params: dict[str, float]

    # Per-episode metrics for Fig-2d-style error bars (one entry per test episode)
    test_per_episode_kl: list[float] = field(default_factory=list)
    test_per_episode_acc: list[float] = field(default_factory=list)

    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.model_name} ({self.n_params}p): "
            f"CE test={self.test_ce:.1f}  KL test={self.test_kl:.2f}  "
            f"action_acc test={self.test_acc:.3f}  BIC={self.bic:.1f}"
        )


def bic_score(train_loss_per_block: float, n_params: int, n_episodes: int, n_predict_trials: int) -> float:
    """BIC = 2·loss_total + k·log(N_obs). loss_total = per_block·n_episodes; N_obs = episodes·trials."""
    loss_total = train_loss_per_block * n_episodes
    n_obs = n_episodes * n_predict_trials
    return 2 * loss_total + n_params * math.log(n_obs)
