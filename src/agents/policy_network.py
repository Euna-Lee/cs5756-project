"""Actor-critic neural network for Pokémon battle decisions.

Architecture
------------
A shared MLP trunk feeds into two heads:

* **Actor head** – outputs a logit vector of size ``ACTION_DIM`` (9).
  During inference, invalid actions are masked to -inf before softmax.
* **Critic head** – outputs a scalar state-value estimate V(s).

The network supports both a vanilla (deterministic) policy (``predict``) and
a stochastic policy suitable for PPO (``evaluate_actions``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.environment.obs_encoder import OBS_DIM
from src.environment.action_space import ACTION_DIM


class PolicyNetwork(nn.Module):
    """Shared-trunk actor-critic MLP.

    Parameters
    ----------
    obs_dim:
        Dimension of the observation vector (default: OBS_DIM = 384).
    action_dim:
        Number of discrete actions (default: ACTION_DIM = 9).
    hidden_sizes:
        Sizes of hidden layers in the shared trunk.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: List[int] = [256, 256],
    ) -> None:
        super().__init__()

        # Shared trunk
        layers: List[nn.Module] = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.LayerNorm(h), nn.ReLU()]
            in_size = h
        self.trunk = nn.Sequential(*layers)

        # Actor head
        self.actor_head = nn.Linear(in_size, action_dim)

        # Critic head
        self.critic_head = nn.Linear(in_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller output scale for the policy head
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, values).

        Parameters
        ----------
        obs:
            Float tensor of shape (batch, obs_dim).
        action_mask:
            Optional boolean tensor of shape (batch, action_dim).
            True = valid action.  Invalid actions are set to -1e9 before
            computing the distribution.

        Returns
        -------
        logits : (batch, action_dim)
        values : (batch,)
        """
        features = self.trunk(obs)
        logits = self.actor_head(features)
        values = self.critic_head(features).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        return logits, values

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or greedily select) an action.

        Returns
        -------
        action      : (batch,)  int64
        log_prob    : (batch,)  float32
        value       : (batch,)  float32
        """
        logits, values = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities, entropy, and values for given (obs, action) pairs.

        Used during the PPO update step.

        Returns
        -------
        log_probs : (batch,)
        entropy   : scalar
        values    : (batch,)
        """
        logits, values = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy, values
