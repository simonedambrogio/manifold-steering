"""Best RL: 6-parameter cognitive model (NHB / hybrid-rnn `CogMod`).

Extends `SimpleRL` with **forgetting toward init**, **persistent perseveration** (chosen-arm
bias that decays with forgetting), **transient perseveration** (one-trial choice bonus to
the just-chosen arm), and a **fittable initial value**. Per hybrid-rnn's `CogMod.__call__`:

    new_value = prev_value + action · α · (r - prev_value·action)        # delta on chosen
    new_value = (1 - forget) · new_value + forget · init_value          # decay to init
    new_value = new_value + persev_p · action                            # persistent bias
    pers_value = new_value + persev_t · last_action                      # transient bias (one-trial)
    P(a) = softmax(β · pers_value)

State carries `(value, last_action_onehot)` — the latter so the transient perseveration
in the *next* trial's policy bonuses the arm just chosen. Bounded params reparameterized
through sigmoid/softplus/tanh; `init_value` is free (real-valued).

Pearce-Hall variable α (an optional extension noted in the paper Methods) is *not* in this
class — when needed, a `BestRLVarAlpha` subclass will add α_t to the state and the
`α_{t+1} = w·|δ| + (1-w)·α_t` update.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from cogmod.base import (
    BehavioralModel,
    positive_param,
    real_param,
    signed_unit_param,
    unit_param,
)


class BestRL(BehavioralModel):
    n_arms = 4

    def __init__(
        self,
        alpha_init: float = 0.3,
        beta_init: float = 0.1,
        forget_init: float = 0.05,
        init_value_init: float = 50.0,
    ):
        super().__init__()
        self._alpha_raw = unit_param(alpha_init)             # sigmoid → (0, 1)
        self._beta_raw = positive_param(beta_init)           # softplus → (0, ∞)
        self._forget_raw = unit_param(forget_init)           # sigmoid → (0, 1)
        self._persev_p_raw = signed_unit_param(0.0)          # tanh → (-1, 1)
        self._persev_t_raw = signed_unit_param(0.0)          # tanh → (-1, 1)
        self._init_value = real_param(init_value_init)       # unbounded

    # ----- transformed parameters -----
    @property
    def alpha(self) -> torch.Tensor:    return torch.sigmoid(self._alpha_raw)
    @property
    def beta(self) -> torch.Tensor:     return F.softplus(self._beta_raw)
    @property
    def forget(self) -> torch.Tensor:   return torch.sigmoid(self._forget_raw)
    @property
    def persev_p(self) -> torch.Tensor: return torch.tanh(self._persev_p_raw)
    @property
    def persev_t(self) -> torch.Tensor: return torch.tanh(self._persev_t_raw)
    @property
    def init_value(self) -> torch.Tensor: return self._init_value

    # ----- BehavioralModel interface -----
    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._alpha_raw.device
        value = self.init_value * torch.ones(
            (batch_size, self.n_arms), dtype=torch.float32, device=device
        )
        last_action = torch.zeros(
            (batch_size, self.n_arms), dtype=torch.float32, device=device
        )
        return (value, last_action)

    def policy(self, state) -> torch.Tensor:
        value, last_action = state
        pers_value = value + self.persev_t * last_action   # transient perseveration
        return F.softmax(self.beta * pers_value, dim=-1)

    def update(self, state, action_onehot: torch.Tensor, reward: torch.Tensor):
        value, _ = state
        # Delta-rule update on the chosen arm
        chosen_q = (value * action_onehot).sum(dim=-1, keepdim=True)
        rpe = reward.unsqueeze(-1) - chosen_q
        new_value = value + action_onehot * self.alpha * rpe
        # Forget toward init (applies to all arms — chosen and unchosen alike)
        new_value = (1 - self.forget) * new_value + self.forget * self.init_value
        # Persistent perseveration: chosen arm gets a stored bonus (decays via forget next step)
        new_value = new_value + self.persev_p * action_onehot
        return (new_value, action_onehot)

    def _value_of(self, state) -> torch.Tensor:
        """For `value_trajectory`: extract the value tensor from the state tuple."""
        return state[0]

    def params_dict(self) -> dict[str, float]:
        return {
            "alpha":      self.alpha.detach().item(),
            "beta":       self.beta.detach().item(),
            "forget":     self.forget.detach().item(),
            "persev_p":   self.persev_p.detach().item(),
            "persev_t":   self.persev_t.detach().item(),
            "init_value": self.init_value.detach().item(),
        }
