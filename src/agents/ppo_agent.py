"""Proximal Policy Optimization (PPO) agent for Pokémon battles.

The implementation follows the original PPO paper (Schulman et al., 2017)
with the following standard additions:

* Generalized Advantage Estimation (GAE, Schulman et al., 2015)
* Value-function clipping
* Entropy bonus for exploration
* Gradient norm clipping

The agent wraps a ``PolicyNetwork`` and exposes a simple ``act`` / ``update``
interface that the RL training loop can call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.policy_network import PolicyNetwork
from src.environment.obs_encoder import OBS_DIM
from src.environment.action_space import ACTION_DIM


@dataclass
class PPOConfig:
    """Hyper-parameters for PPO training."""

    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])

    # Rollout
    n_steps: int = 2048        # steps per update
    n_envs: int = 1            # parallel environments (currently 1)

    # PPO objective
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Optimiser
    learning_rate: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64

    # Devices
    device: str = "cpu"


@dataclass
class RolloutTransition:
    obs: np.ndarray
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    action_mask: Optional[np.ndarray] = None


class RolloutBuffer:
    """Stores one rollout and computes GAE advantages."""

    def __init__(self, n_steps: int, obs_dim: int, gamma: float, gae_lambda: float) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.action_masks: List[Optional[np.ndarray]] = []

    def add(self, transition: RolloutTransition) -> None:
        self.observations.append(transition.obs)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.dones.append(transition.done)
        self.log_probs.append(transition.log_prob)
        self.values.append(transition.value)
        self.action_masks.append(transition.action_mask)

    def __len__(self) -> int:
        return len(self.observations)

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """Compute GAE advantages and discounted returns in-place."""
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def get_tensors(
        self, device: str
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        obs_t = torch.tensor(np.array(self.observations), dtype=torch.float32, device=device)
        acts_t = torch.tensor(self.actions, dtype=torch.long, device=device)
        log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        adv_t = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(self.returns, dtype=torch.float32, device=device)

        if any(m is not None for m in self.action_masks):
            masks = np.array(
                [m if m is not None else np.ones(ACTION_DIM, dtype=bool)
                 for m in self.action_masks]
            )
            masks_t: Optional[torch.Tensor] = torch.tensor(masks, dtype=torch.bool, device=device)
        else:
            masks_t = None

        return obs_t, acts_t, log_probs_t, adv_t, ret_t, masks_t


class PPOAgent:
    """PPO agent wrapping a ``PolicyNetwork``.

    Parameters
    ----------
    config:
        PPO hyper-parameters.
    pretrained_path:
        Optional path to a ``.pt`` file containing a state-dict for
        behaviour-cloning warm-start.
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        pretrained_path: Optional[str] = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)

        self.policy = PolicyNetwork(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=False)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

        self.buffer = RolloutBuffer(
            n_steps=self.config.n_steps,
            obs_dim=OBS_DIM,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Choose an action for a single observation.

        Returns
        -------
        action    : int
        log_prob  : float
        value     : float
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = None
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        action_t, log_prob_t, value_t = self.policy.get_action(obs_t, mask_t, deterministic)
        return int(action_t.item()), float(log_prob_t.item()), float(value_t.item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """Run one PPO update on the current rollout buffer.

        Returns
        -------
        dict with scalar loss metrics.
        """
        cfg = self.config
        obs_t, acts_t, old_log_probs_t, adv_t, ret_t, masks_t = self.buffer.get_tensors(
            str(self.device)
        )

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_t)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        for _ in range(cfg.n_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.batch_size):
                idx = indices[start : start + cfg.batch_size]
                batch_obs = obs_t[idx]
                batch_acts = acts_t[idx]
                batch_old_lp = old_log_probs_t[idx]
                batch_adv = adv_t[idx]
                batch_ret = ret_t[idx]
                batch_masks = masks_t[idx] if masks_t is not None else None

                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    batch_obs, batch_acts, batch_masks
                )

                # Ratio and clipped objective
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with optional clipping
                value_loss = nn.functional.mse_loss(new_values, batch_ret)

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1

        self.buffer.clear()
        denom = max(n_batches, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
        }

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)
