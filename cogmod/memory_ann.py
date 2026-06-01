"""Memory-ANN: hybrid model — cognitive scaffold + learned RNN updates.

Port of hybrid-rnn `bi_rnn.py` with `s=True` baked in for both modules (true recurrence),
and `w_v`, `w_h` exposed as fittable scalars (released code fixes them at 0.5 each).
Two parallel RNN modules:

  Value module — input = [chosen_value, reward, v_state]; produces a *scalar* update
    applied only to the chosen arm. Forgetting toward init_value applied to all arms
    before the update.

      v_state_t = tanh( Linear([Q_{t-1}(a_{t-1}) ⊕ r_{t-1} ⊕ v_state_{t-1}]) )
      update_t  = Linear(v_state_t)
      Q_t       = (1−forget)·Q_{t-1} + forget·init_v
      Q_t[a_{t-1}] += update_t

  Habit module — input = [action, h_state]; produces per-arm habit (no reward input —
    pure action-history dynamics).

      h_state_t = tanh( Linear([a_{t-1}_onehot ⊕ h_state_{t-1}]) )
      habit_t   = Linear(h_state_t)

  Combine + choice: P(a) = softmax( β · (w_v · Q + w_h · habit) ).

~800–1000 params total. Trainable: the four `Linear` layers + `β`, `forget`, `w_v`, `w_h`,
`init_value_v`, `init_value_h` (bounded params reparameterized through softplus/sigmoid).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cogmod.base import BehavioralModel, positive_param, real_param, unit_param


class MemoryANN(BehavioralModel):
    n_arms = 4

    def __init__(
        self,
        hidden_size: int = 16,
        n_arms: int = 4,
        init_value: float = 0.5,
    ):
        super().__init__()
        self.n_arms = n_arms
        self.hidden_size = hidden_size

        # Value module: [chosen_value(1) + reward(1) + v_state(H)] -> v_state(H), then update(1)
        self.v_fc = nn.Linear(1 + 1 + hidden_size, hidden_size)
        self.v_update = nn.Linear(hidden_size, 1)

        # Habit module: [action_onehot(n) + h_state(H)] -> h_state(H), then habit(n)
        self.h_fc = nn.Linear(n_arms + hidden_size, hidden_size)
        self.h_readout = nn.Linear(hidden_size, n_arms)

        # Fittable scalars
        self._beta_raw = positive_param(0.1)
        self._forget_raw = unit_param(0.05)
        self._w_v_raw = unit_param(0.5)
        self._w_h_raw = unit_param(0.5)
        self._init_value_v = real_param(init_value)
        self._init_value_h = real_param(init_value)

    # ----- transformed parameters -----
    @property
    def beta(self) -> torch.Tensor:   return F.softplus(self._beta_raw)
    @property
    def forget(self) -> torch.Tensor: return torch.sigmoid(self._forget_raw)
    @property
    def w_v(self) -> torch.Tensor:    return torch.sigmoid(self._w_v_raw)
    @property
    def w_h(self) -> torch.Tensor:    return torch.sigmoid(self._w_h_raw)

    # ----- BehavioralModel interface -----
    def initial_state(self, batch_size: int):
        device = self.v_fc.weight.device
        zero = torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=device)
        ones_n = torch.ones((batch_size, self.n_arms), dtype=torch.float32, device=device)
        return (
            zero.clone(),                       # h_state
            zero.clone(),                       # v_state
            self._init_value_h * ones_n,        # habit
            self._init_value_v * ones_n,        # value
        )

    def policy(self, state) -> torch.Tensor:
        _, _, habit, value = state
        hv_combo = self.w_v * value + self.w_h * habit
        return F.softmax(self.beta * hv_combo, dim=-1)

    def update(self, state, action_onehot: torch.Tensor, reward: torch.Tensor):
        h_state, v_state, habit, value = state

        # Value module
        chosen_value = (value * action_onehot).sum(dim=-1, keepdim=True)            # [B, 1]
        v_inp = torch.cat([chosen_value, reward.unsqueeze(-1), v_state], dim=-1)
        next_v_state = torch.tanh(self.v_fc(v_inp))
        update_val = self.v_update(next_v_state)                                    # [B, 1]
        value_decay = (1 - self.forget) * value + self.forget * self._init_value_v
        next_value = value_decay + action_onehot * update_val

        # Habit module
        h_inp = torch.cat([action_onehot, h_state], dim=-1)
        next_h_state = torch.tanh(self.h_fc(h_inp))
        next_habit = self.h_readout(next_h_state)

        return (next_h_state, next_v_state, next_habit, next_value)

    def _value_of(self, state) -> torch.Tensor:
        """Combined value+habit — what drives choice."""
        _, _, habit, value = state
        return self.w_v * value + self.w_h * habit

    def params_dict(self) -> dict[str, float | int]:
        return {
            "beta":         self.beta.detach().item(),
            "forget":       self.forget.detach().item(),
            "w_v":          self.w_v.detach().item(),
            "w_h":          self.w_h.detach().item(),
            "init_value_v": self._init_value_v.detach().item(),
            "init_value_h": self._init_value_h.detach().item(),
            "hidden_size":  self.hidden_size,
            "n_trainable_params": sum(p.numel() for p in self.parameters()),
        }
