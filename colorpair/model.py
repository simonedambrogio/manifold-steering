"""The color-pair comparison network -- Option B (the "fair" architecture).

Two learned parts, no geometric prior:
  - a **shared encoder** phi: one-hot color (N) -> R^k. The same phi embeds every color in every slot,
    so there is one consistent color geometry to measure and steer. We never tell it colors are circular;
    the one-hot input asserts no order at all.
  - a **learned comparison** g: (phi_a, phi_b) -> a scalar "distance". We do *not* hard-code Euclidean
    distance (that would be Option A and would bake in the ring). g is free to compute distance however
    it likes; circularity must be discovered. g is symmetrized over color order (the true task is
    order-invariant within a pair).

Choice: logit = g(pair1) - g(pair0); P(pair 0 is the closer pair) = sigmoid(logit). Trained with BCE.

The residual risk of a fully-learned g is that the ring could hide in g rather than phi. We block the
lazy lookup solution with held-out pairs + a larger N (see `env.make_split`): to generalize, the model
must put the ring metric where it is reusable -- in phi. We then measure phi and find out. A `k=1`
encoder cannot fold (colors live on a line), so it must fail the seam pairs: the necessity control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorPairNet(nn.Module):
    def __init__(self, n_colors: int, k: int = 8, enc_hidden: int = 64, cmp_hidden: int = 64):
        super().__init__()
        self.n_colors = n_colors
        self.k = k
        # shared encoder phi: one-hot(N) -> R^k
        self.phi = nn.Sequential(
            nn.Linear(n_colors, enc_hidden), nn.ReLU(),
            nn.Linear(enc_hidden, k),
        )
        # learned comparison g: R^{2k} -> R (a scalar distance), symmetrized over the two color orders
        self.g = nn.Sequential(
            nn.Linear(2 * k, cmp_hidden), nn.ReLU(),
            nn.Linear(cmp_hidden, cmp_hidden), nn.ReLU(),
            nn.Linear(cmp_hidden, 1),
        )

    def embed(self, colors: torch.Tensor) -> torch.Tensor:
        """Embed integer color indices (any shape [...]) -> [..., k] via one-hot input + phi."""
        oh = F.one_hot(colors, self.n_colors).float()
        return self.phi(oh)

    def embed_all(self, device=None) -> torch.Tensor:
        """phi for every color -> [N, k] (the object we test for a ring and steer)."""
        idx = torch.arange(self.n_colors, device=device or next(self.parameters()).device)
        return self.embed(idx)

    def pair_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Symmetric learned distance g for embedding tensors a, b ([..., k]) -> [...]."""
        d_ab = self.g(torch.cat([a, b], dim=-1))
        d_ba = self.g(torch.cat([b, a], dim=-1))
        return 0.5 * (d_ab + d_ba).squeeze(-1)

    def logits_from_embeddings(self, emb: torch.Tensor) -> torch.Tensor:
        """Choice logit from per-slot embeddings emb [B, 4, k]. logit > 0 => pair 0 is closer.

        This path takes embeddings directly so the steering experiment can inject a modified phi for
        one color while leaving the others at their natural values."""
        d0 = self.pair_distance(emb[:, 0], emb[:, 1])
        d1 = self.pair_distance(emb[:, 2], emb[:, 3])
        return d1 - d0

    def forward(self, colors: torch.Tensor) -> torch.Tensor:
        """colors [B, 4] integer indices -> choice logit [B]."""
        return self.logits_from_embeddings(self.embed(colors))
