"""Reinforcement learning training loop (PPO).

Supports two modes:
  1. **Scratch** – train from randomly initialised weights.
  2. **BC warmstart** – initialise from a behaviour-cloning checkpoint then
     fine-tune with PPO.

The trainer wraps a gymnasium ``Env`` (either the real ``PokemonBattleEnv``
or the lightweight ``MockBattleEnv`` for testing) and drives the standard
PPO collect-then-update loop.

Metrics logged per update:
  - Episode reward (mean, std)
  - Episode length (mean)
  - Win rate (fraction of episodes with positive terminal reward)
  - PPO losses (policy, value, entropy)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.agents.ppo_agent import PPOAgent, PPOConfig, RolloutTransition
from src.environment.obs_encoder import OBS_DIM
from src.environment.action_space import ACTION_DIM, get_action_mask


@dataclass
class RLConfig:
    """Top-level RL training configuration."""

    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Total training budget
    total_timesteps: int = 500_000

    # Logging / checkpointing
    log_interval: int = 10        # log every N updates
    save_interval: int = 50       # save checkpoint every N updates
    save_dir: str = "checkpoints"
    run_name: str = "rl_run"

    # BC warmstart
    bc_checkpoint: Optional[str] = None  # path to BC .pt file


class RLTrainer:
    """PPO trainer for Pokémon battles.

    Parameters
    ----------
    env_fn:
        Zero-argument callable that returns a gymnasium ``Env``.
    config:
        Training configuration.
    """

    def __init__(
        self,
        env_fn: Callable,
        config: Optional[RLConfig] = None,
    ) -> None:
        self.config = config or RLConfig()
        self.env = env_fn()

        ppo_cfg = self.config.ppo
        if self.config.bc_checkpoint:
            agent = PPOAgent(ppo_cfg, pretrained_path=self.config.bc_checkpoint)
        else:
            agent = PPOAgent(ppo_cfg)
        self.agent = agent

        self.history: List[Dict] = []
        os.makedirs(self.config.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> List[Dict]:
        """Run PPO training for ``total_timesteps`` environment steps.

        Returns
        -------
        List of per-update metric dicts.
        """
        cfg = self.config
        ppo_cfg = cfg.ppo
        agent = self.agent
        env = self.env

        obs, _ = env.reset()
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        episode_wins: List[bool] = []

        current_ep_reward = 0.0
        current_ep_length = 0

        total_steps = 0
        update_count = 0

        target_steps = cfg.total_timesteps

        while total_steps < target_steps:
            # --- Collect rollout ---
            for _ in range(ppo_cfg.n_steps):
                mask = _get_mask_from_env(env)
                action, log_prob, value = agent.act(obs, action_mask=mask)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.buffer.add(
                    RolloutTransition(
                        obs=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        log_prob=log_prob,
                        value=value,
                        action_mask=mask,
                    )
                )

                obs = next_obs
                current_ep_reward += reward
                current_ep_length += 1
                total_steps += 1

                if done:
                    episode_rewards.append(current_ep_reward)
                    episode_lengths.append(current_ep_length)
                    episode_wins.append(current_ep_reward > 0)
                    current_ep_reward = 0.0
                    current_ep_length = 0
                    obs, _ = env.reset()

                if total_steps >= target_steps:
                    break

            # --- Compute advantages ---
            last_value = 0.0
            if not done:
                _, _, last_value = agent.act(obs)

            agent.buffer.compute_returns_and_advantages(last_value)

            # --- PPO update ---
            losses = agent.update()
            update_count += 1

            # --- Logging ---
            if update_count % cfg.log_interval == 0 and episode_rewards:
                n = len(episode_rewards)
                metrics = {
                    "update": update_count,
                    "total_steps": total_steps,
                    "mean_ep_reward": float(np.mean(episode_rewards[-100:])),
                    "std_ep_reward": float(np.std(episode_rewards[-100:])),
                    "mean_ep_length": float(np.mean(episode_lengths[-100:])),
                    "win_rate": float(np.mean(episode_wins[-100:])),
                    **losses,
                }
                self.history.append(metrics)
                print(
                    f"Update {update_count:4d} | "
                    f"steps {total_steps:7d} | "
                    f"rew {metrics['mean_ep_reward']:+.2f} | "
                    f"win {metrics['win_rate']:.2%} | "
                    f"pl {metrics['policy_loss']:.4f} | "
                    f"vl {metrics['value_loss']:.4f}"
                )

            # --- Checkpoint ---
            if update_count % cfg.save_interval == 0:
                ckpt_path = os.path.join(
                    cfg.save_dir, f"{cfg.run_name}_step{total_steps}.pt"
                )
                agent.save(ckpt_path)

        # Final checkpoint
        final_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_final.pt")
        agent.save(final_path)

        return self.history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mask_from_env(env) -> Optional[np.ndarray]:
    """Try to get an action mask from the environment, if supported."""
    if hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    # Unwrap poke-env battle to extract mask
    battle = getattr(env, "_current_battle", None)
    if battle is not None:
        return np.array(get_action_mask(battle), dtype=bool)
    return None
