"""Battle logging utilities.

Serialises per-step data (observation, action, reward, battle metadata) to a
JSONL file for later analysis and behaviour-cloning data collection.

Each line in the JSONL file is a JSON object with the following fields:

  battle_id    : str   – unique identifier for the battle
  step         : int   – turn number within the battle
  observation  : list  – float vector of length OBS_DIM
  action       : int   – chosen action index
  reward       : float – shaped reward at this step
  done         : bool  – True if the battle ended on this step
  outcome      : str | null – "win" | "loss" | "draw" | null (non-terminal)
  info         : dict  – arbitrary extra info (e.g. move name, switch target)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np


class BattleLogger:
    """Logs battle trajectories to a JSONL file.

    Parameters
    ----------
    log_dir:
        Directory where JSONL files are written.
    filename:
        Output filename (inside ``log_dir``).  Defaults to a timestamped name.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        filename: Optional[str] = None,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"battles_{ts}.jsonl"
        self._path = os.path.join(log_dir, filename)
        self._file = open(self._path, "a")

        # Per-episode accumulators
        self._current_battle_id: Optional[str] = None
        self._step: int = 0
        self._episode_rewards: List[float] = []

        # Aggregate statistics
        self.total_battles: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.total_draws: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_battle(self, battle_id: Optional[str] = None) -> str:
        """Start logging a new battle.  Returns the battle_id."""
        self._current_battle_id = battle_id or str(uuid.uuid4())[:8]
        self._step = 0
        self._episode_rewards = []
        return self._current_battle_id

    def log_step(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        outcome: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single environment step.

        Parameters
        ----------
        observation:
            Encoded observation vector (numpy array).
        action:
            Chosen action index.
        reward:
            Shaped reward for this step.
        done:
            Whether the battle ended.
        outcome:
            "win", "loss", or "draw" (only meaningful when done=True).
        info:
            Extra metadata dict.
        """
        record = {
            "battle_id": self._current_battle_id,
            "step": self._step,
            "observation": observation.tolist(),
            "action": action,
            "reward": float(reward),
            "done": done,
            "outcome": outcome if done else None,
            "info": info or {},
        }
        self._file.write(json.dumps(record) + "\n")
        self._episode_rewards.append(float(reward))
        self._step += 1

        if done:
            self._finalise_battle(outcome)

    def end_battle(self, outcome: str) -> None:
        """Explicitly close a battle with a given outcome.

        Useful when the environment signals termination outside of ``log_step``.
        """
        self._finalise_battle(outcome)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics."""
        total = self.total_battles
        return {
            "total_battles": total,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "draws": self.total_draws,
            "win_rate": self.total_wins / total if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalise_battle(self, outcome: Optional[str]) -> None:
        self.total_battles += 1
        if outcome == "win":
            self.total_wins += 1
        elif outcome == "loss":
            self.total_losses += 1
        else:
            self.total_draws += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_demonstrations(path: str) -> List[Dict]:
    """Load a JSONL log file and return a list of step records.

    Filters out records that are missing required keys.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "observation" in rec and "action" in rec:
                records.append(rec)
    return records


def compute_battle_stats(log_path: str) -> Dict[str, Any]:
    """Compute summary statistics from a JSONL battle log.

    Returns win rate, average episode length, average total reward, etc.
    """
    records = load_demonstrations(log_path)
    if not records:
        return {"error": "no records"}

    battles: Dict[str, Dict] = {}
    for rec in records:
        bid = rec["battle_id"]
        if bid not in battles:
            battles[bid] = {"rewards": [], "outcome": None}
        battles[bid]["rewards"].append(rec["reward"])
        if rec.get("done"):
            battles[bid]["outcome"] = rec.get("outcome")

    outcomes = [b["outcome"] for b in battles.values()]
    total_rewards = [sum(b["rewards"]) for b in battles.values()]
    ep_lengths = [len(b["rewards"]) for b in battles.values()]
    n = len(battles)
    wins = sum(1 for o in outcomes if o == "win")

    return {
        "n_battles": n,
        "win_rate": wins / n if n > 0 else 0.0,
        "mean_ep_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "outcomes": {
            "win": wins,
            "loss": sum(1 for o in outcomes if o == "loss"),
            "draw": sum(1 for o in outcomes if o not in ("win", "loss")),
        },
    }
