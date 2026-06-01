"""Vanilla RNN: small recurrent network with no cognitive scaffold.

Port of hybrid-rnn `rnn.py` with `s=True` baked in (true recurrence — the released
default `s=False` makes it a per-trial MLP, which is degenerate). Hidden state evolves
across trials from `(action, reward)` inputs; choice logits read out from the hidden
state. Tiny: hidden_size = 16 by default (matches NHB's `network_params.hidden_size`).

    h_t = tanh( Linear( [action_{t-1}_onehot ⊕ reward_{t-1} ⊕ h_{t-1}] ) )
    P(a_t) = softmax( Linear(h_t) )

Trainable: the two `Linear` layers' weights and biases. ~400 params at hidden_size=16.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cogmod.base import BehavioralModel


class VanillaRNN(BehavioralModel):
    n_arms = 4

    def __init__(self, hidden_size: int = 16, n_arms: int = 4):
        super().__init__()
        self.n_arms = n_arms
        self.hidden_size = hidden_size
        # Update: [action_onehot (n) + reward (1) + hidden (H)] -> hidden (H)
        self.update_fc = nn.Linear(n_arms + 1 + hidden_size, hidden_size)
        # Readout: hidden (H) -> action logits (n)
        self.readout = nn.Linear(hidden_size, n_arms)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        device = self.update_fc.weight.device
        return torch.zeros(
            (batch_size, self.hidden_size), dtype=torch.float32, device=device
        )

    def policy(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.readout(state), dim=-1)

    def update(
        self, state: torch.Tensor, action_onehot: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        inp = torch.cat([action_onehot, reward.unsqueeze(-1), state], dim=-1)
        return torch.tanh(self.update_fc(inp))

    def _value_of(self, state: torch.Tensor) -> torch.Tensor:
        """Action logits — what determines the choice (the analog of per-arm value)."""
        return self.readout(state)

    def params_dict(self) -> dict[str, float | int]:
        return {
            "hidden_size": self.hidden_size,
            "n_trainable_params": sum(p.numel() for p in self.parameters()),
        }
