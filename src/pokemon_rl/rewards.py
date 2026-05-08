from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """
    Week-1 reward config.

    Default is sparse terminal reward (matches your hypothesis focus).
    """

    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    terminal_tie: float = 0.0


def terminal_reward(battle: Any, cfg: RewardConfig | None = None) -> float:
    """
    Terminal-only reward:
    - +1 if agent won
    - -1 if agent lost
    - 0 otherwise (including mid-episode or ties/unknown)
    """
    cfg = cfg or RewardConfig()
    finished = getattr(battle, "finished", False)
    if not finished:
        return 0.0
    won = getattr(battle, "won", None)
    if won is True:
        return float(cfg.terminal_win)
    if won is False:
        return float(cfg.terminal_loss)
    return float(cfg.terminal_tie)

