#!/usr/bin/env python3
"""Collect expert demonstrations from the SimpleHeuristicsPlayer.

The script runs N self-play battles where the SimpleHeuristicsPlayer controls
both sides, serialising each step as a JSONL record for behaviour cloning.

Usage
-----
    python scripts/collect_demonstrations.py \\
        --n-battles 1000 \\
        --output-path data/expert_demos.jsonl \\
        --format gen8randombattle

Notes
-----
Running against a live Pokémon Showdown server requires the server to be
running locally.  See the README for setup instructions.

For offline / CI testing, pass ``--mock`` to use a ``MockBattleEnv`` and
generate synthetic demonstration data without a server.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

# Ensure src is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.expert_bot import SimpleHeuristicsPlayer
from src.environment.obs_encoder import encode_battle, OBS_DIM
from src.environment.action_space import ACTION_DIM, get_action_mask
from src.utils.battle_logger import BattleLogger


def collect_mock_demonstrations(
    n_battles: int,
    output_path: str,
    seed: int = 42,
) -> None:
    """Generate synthetic demonstrations using random observations and the
    SimpleHeuristicsPlayer heuristic on mock battle objects (no server needed).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    expert = SimpleHeuristicsPlayer()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    written = 0
    with open(output_path, "w") as f:
        for battle_idx in range(n_battles):
            # Simulate a short battle (10-40 turns)
            n_turns = rng.randint(10, 40)
            battle_id = f"mock_{battle_idx:05d}"

            for step in range(n_turns):
                obs = np.random.uniform(-1, 1, size=OBS_DIM).astype(np.float32)
                mock_battle = _make_mock_battle(rng)
                action = expert.choose_action(mock_battle)
                action_mask = get_action_mask(mock_battle)

                done = step == n_turns - 1
                outcome = "win" if rng.random() > 0.5 else "loss"

                record = {
                    "battle_id": battle_id,
                    "step": step,
                    "observation": obs.tolist(),
                    "action": action,
                    "action_mask": action_mask,
                    "reward": 1.0 if (done and outcome == "win") else -1.0 if done else 0.0,
                    "done": done,
                    "outcome": outcome if done else None,
                }
                f.write(json.dumps(record) + "\n")
                written += 1

    print(f"Wrote {written} steps from {n_battles} mock battles → {output_path}")


def collect_live_demonstrations(
    n_battles: int,
    output_path: str,
    battle_format: str = "gen8randombattle",
) -> None:  # pragma: no cover
    """Collect demonstrations from a live Pokémon Showdown server.

    Requires poke-env and a running server.
    """
    try:
        import asyncio
        from poke_env.player import SimpleHeuristicsPlayer as PokeSimpleHeuristics
    except ImportError:
        print("poke-env is not installed.  Run: pip install poke-env")
        sys.exit(1)

    raise NotImplementedError(
        "Live demonstration collection is not yet implemented.  "
        "Use --mock for offline testing."
    )


def _make_mock_battle(rng: random.Random) -> dict:
    """Create a minimal mock battle dict for the heuristic bot."""
    types = ["fire", "water", "grass", "electric", "psychic", "dark", "steel", "dragon"]
    n_moves = rng.randint(1, 4)
    n_switches = rng.randint(0, 5)

    def make_pokemon(alive: bool = True):
        return {
            "species": f"pokemon_{rng.randint(0, 150)}",
            "current_hp_fraction": rng.uniform(0.0, 1.0) if alive else 0.0,
            "fainted": not alive,
            "status": None,
            "type_1": rng.choice(types),
            "type_2": rng.choice(types) if rng.random() > 0.5 else None,
            "base_stats": {
                "hp": rng.randint(40, 255),
                "atk": rng.randint(40, 165),
                "def": rng.randint(40, 165),
                "spa": rng.randint(40, 165),
                "spd": rng.randint(40, 165),
                "spe": rng.randint(40, 165),
            },
        }

    def make_move():
        return {
            "type": rng.choice(types),
            "base_power": rng.choice([0, 40, 60, 80, 90, 100, 120]),
            "accuracy": rng.uniform(0.7, 1.0),
            "category": rng.choice(["physical", "special", "status"]),
            "current_pp": rng.randint(1, 16),
            "max_pp": 16,
        }

    active = make_pokemon(alive=True)
    opp_active = make_pokemon(alive=True)
    switches = [make_pokemon() for _ in range(n_switches)]
    moves = [make_move() for _ in range(n_moves)]

    return {
        "active_pokemon": active,
        "available_moves": moves,
        "available_switches": switches,
        "force_switch": False,
        "trapped": False,
        "opponent_active_pokemon": opp_active,
        "team": {f"slot_{i}": make_pokemon(i < 4) for i in range(6)},
        "opponent_team": {f"opp_{i}": make_pokemon(i < 4) for i in range(6)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect expert Pokémon battle demonstrations.")
    parser.add_argument("--n-battles", type=int, default=500)
    parser.add_argument("--output-path", type=str, default="data/expert_demos.jsonl")
    parser.add_argument("--format", type=str, default="gen8randombattle")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Use mock battles (no server required)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mock:
        collect_mock_demonstrations(args.n_battles, args.output_path, seed=args.seed)
    else:
        collect_live_demonstrations(args.n_battles, args.output_path, args.format)


if __name__ == "__main__":
    main()
