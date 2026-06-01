"""Simple RL: 2-parameter Q-learner with softmax choice rule.

The simplest cognitive baseline: a tabular Q-learner that updates only the chosen arm
by a delta rule, with a softmax (inverse-temperature) choice rule. Two fittable params,
`alpha` (learning rate) and `beta` (inverse temperature):

    Q_{t+1}(a_t) = Q_t(a_t) + α · (r_t - Q_t(a_t))   (delta rule, chosen arm only)
    P(a)         = softmax(β · Q_t)                    (choice rule)

Initial value `Q_init = 50` is fixed (mean of the [1, 100] reward range). Bounded params
are reparameterized through `sigmoid` (α) and `softplus` (β), so optimization is
unconstrained — the same trick hybrid-rnn's `CogMod` uses.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from cogmod.base import BehavioralModel, positive_param, unit_param


class SimpleRL(BehavioralModel):
    n_arms = 4

    def __init__(self, q_init: float = 50.0, alpha_init: float = 0.3, beta_init: float = 0.1):
        super().__init__()
        self.q_init = q_init
        self._alpha_raw = unit_param(alpha_init)       # sigmoid → (0, 1)
        self._beta_raw = positive_param(beta_init)     # softplus → (0, ∞)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size, self.n_arms), self.q_init,
            dtype=torch.float32, device=self._alpha_raw.device,
        )

    def policy(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.beta * state, dim=-1)

    def update(self, state: torch.Tensor, action_onehot: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        # delta rule on the chosen arm: Q[a] += α · (r - Q[a])
        chosen_q = (state * action_onehot).sum(dim=-1, keepdim=True)
        rpe = reward.unsqueeze(-1) - chosen_q
        return state + action_onehot * self.alpha * rpe

    def params_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha.detach().item(),
            "beta": self.beta.detach().item(),
            "q_init": self.q_init,
        }
