"""Gym-compatible environment wrapper for Pokémon battles.

This module provides ``PokemonBattleEnv``, a ``gymnasium.Env`` subclass that
wraps a poke-env battle player.  The environment can run in two modes:

1. **Self-play** – the agent battles a fixed ``opponent`` player.
2. **Demo / test** – a lightweight ``MockBattleEnv`` with no poke-env dependency
   is provided for unit-testing the observation / reward pipelines.

Reward shaping
--------------
* +30   : win the battle
* -30   : lose the battle
* +2    : opponent Pokémon faints
* -2    : player Pokémon faints
* +Δhp  : proportional HP dealt to opponent (scaled to [0, 1])
* -Δhp  : proportional HP lost by player     (scaled to [0, 1])
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym.spaces import Box, Discrete  # type: ignore

from src.environment.obs_encoder import OBS_DIM, encode_battle
from src.environment.action_space import ACTION_DIM, action_to_move, get_action_mask

# Reward weights
VICTORY_VALUE = 30.0
FAINT_VALUE = 2.0
HP_VALUE = 1.0


class PokemonBattleEnv(gym.Env):
    """Gymnasium wrapper around a poke-env Gen-8 single battle.

    Parameters
    ----------
    player:
        A poke-env Player subclass that will act as the learning agent side.
        The player must expose ``reset()``, ``step(action)`` following the
        poke-env Gen8EnvSinglePlayer interface.
    opponent:
        A poke-env Player used as the opponent (e.g. ``RandomPlayer`` or
        ``SimpleHeuristicsPlayer``).
    battle_format:
        Showdown battle format string, e.g. ``"gen8randombattle"``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        player: Any,
        opponent: Any,
        battle_format: str = "gen8randombattle",
    ) -> None:
        super().__init__()
        self._player = player
        self._opponent = opponent
        self._battle_format = battle_format

        self.observation_space = Box(
            low=np.full(OBS_DIM, -1.0, dtype=np.float32),
            high=np.full(OBS_DIM, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = Discrete(ACTION_DIM)

        self._current_battle: Optional[Any] = None
        self._last_battle_state: Optional[Any] = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._current_battle = self._player.reset()
        self._last_battle_state = self._current_battle
        obs = encode_battle(self._current_battle)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_battle = self._current_battle
        obs_raw, reward, terminated, info = self._player.step(action)
        self._current_battle = self._player.current_battle

        # Convert raw poke-env reward if needed
        if isinstance(reward, (int, float)):
            shaped_reward = float(reward)
        else:
            shaped_reward = self._calc_reward(prev_battle, self._current_battle)

        obs = encode_battle(self._current_battle)
        truncated = False
        return obs, shaped_reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _calc_reward(self, prev: Any, curr: Any) -> float:
        """Compute shaped reward between two consecutive battle snapshots."""
        if curr is None:
            return 0.0

        reward = 0.0

        # Win / loss
        if getattr(curr, "won", None) is True:
            reward += VICTORY_VALUE
        elif getattr(curr, "lost", None) is True:
            reward -= VICTORY_VALUE

        # Fainted Pokémon deltas
        prev_opp_fainted = _count_fainted(getattr(prev, "opponent_team", {}))
        curr_opp_fainted = _count_fainted(getattr(curr, "opponent_team", {}))
        reward += FAINT_VALUE * (curr_opp_fainted - prev_opp_fainted)

        prev_self_fainted = _count_fainted(getattr(prev, "team", {}))
        curr_self_fainted = _count_fainted(getattr(curr, "team", {}))
        reward -= FAINT_VALUE * (curr_self_fainted - prev_self_fainted)

        # HP-based shaping
        prev_opp_hp = _total_hp_fraction(getattr(prev, "opponent_team", {}))
        curr_opp_hp = _total_hp_fraction(getattr(curr, "opponent_team", {}))
        reward += HP_VALUE * (prev_opp_hp - curr_opp_hp)

        prev_self_hp = _total_hp_fraction(getattr(prev, "team", {}))
        curr_self_hp = _total_hp_fraction(getattr(curr, "team", {}))
        reward -= HP_VALUE * (prev_self_hp - curr_self_hp)

        return reward


# ---------------------------------------------------------------------------
# Lightweight mock environment (no poke-env server required)
# ---------------------------------------------------------------------------

class MockBattleEnv(gym.Env):
    """Minimal environment for unit-testing agents and trainers.

    Produces random observations and terminates after ``max_steps`` steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 50) -> None:
        super().__init__()
        self.observation_space = Box(
            low=np.full(OBS_DIM, -1.0, dtype=np.float32),
            high=np.full(OBS_DIM, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = Discrete(ACTION_DIM)
        self._max_steps = max_steps
        self._step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0
        if seed is not None:
            self.observation_space.seed(seed)
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        obs = self.observation_space.sample()
        reward = float(np.random.randn())
        terminated = self._step_count >= self._max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _count_fainted(team: Any) -> int:
    """Count the number of fainted Pokémon in a team dict."""
    if not team:
        return 0
    count = 0
    items = team.values() if isinstance(team, dict) else team
    for poke in items:
        if isinstance(poke, dict):
            if poke.get("fainted", False):
                count += 1
        else:
            if getattr(poke, "fainted", False):
                count += 1
    return count


def _total_hp_fraction(team: Any) -> float:
    """Sum of current HP fractions across a team."""
    if not team:
        return 0.0
    total = 0.0
    items = team.values() if isinstance(team, dict) else team
    for poke in items:
        if isinstance(poke, dict):
            total += poke.get("current_hp_fraction", 0.0)
        else:
            total += getattr(poke, "current_hp_fraction", 0.0)
    return total
