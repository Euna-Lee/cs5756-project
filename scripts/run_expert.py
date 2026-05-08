#!/usr/bin/env python3
"""Run the heuristic expert vs a baseline opponent (quick eval)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pokemon_rl.constants import FORMAT_MAIN
from pokemon_rl.env import LoggedSimpleHeuristicsPlayer, LoggingConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--battle-format", type=str, default=FORMAT_MAIN)
    p.add_argument("--n-battles", type=int, default=10)
    p.add_argument("--max-concurrent-battles", type=int, default=1)
    p.add_argument("--out", type=str, default="data/expert_steps.jsonl")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    async def runner() -> None:
        expert = LoggedSimpleHeuristicsPlayer(
            battle_format=args.battle_format,
            max_concurrent_battles=args.max_concurrent_battles,
            account_configuration=AccountConfiguration.generate("LoggedSimpleHeur-expert", rand=True),
            logging_cfg=LoggingConfig(path=args.out, run_id="run_expert", role="expert"),
        )
        opp = RandomPlayer(
            battle_format=args.battle_format,
            max_concurrent_battles=args.max_concurrent_battles,
            account_configuration=AccountConfiguration.generate("RandomPlayer-opponent", rand=True),
        )
        await expert.battle_against(opp, n_battles=args.n_battles)
        win_rate = expert.n_won_battles / max(1, expert.n_finished_battles)
        print(f"expert vs random: {expert.n_won_battles}/{expert.n_finished_battles} (win_rate={win_rate:.3f})")

    asyncio.run(runner())


if __name__ == "__main__":
    main()
