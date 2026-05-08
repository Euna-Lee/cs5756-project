from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pokemon_rl.representations import ACTION_SPACE_SIZE


@dataclass(frozen=True)
class PolicyOutput:
    logits: torch.Tensor  # (B, A)


class MaskedPolicyNet(nn.Module):
    """
    Simple MLP policy over the fixed 10-action space.

    We apply the legal-action mask by setting illegal logits to a large negative value.
    """

    def __init__(self, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, ACTION_SPACE_SIZE),
        )

    def forward(self, x: torch.Tensor) -> PolicyOutput:
        return PolicyOutput(logits=self.net(x))

    @staticmethod
    def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, A), legal_mask: (B, A) bool
        """
        neg_inf = torch.finfo(logits.dtype).min
        return torch.where(legal_mask, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))

