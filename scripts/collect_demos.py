#!/usr/bin/env python3
"""
Collect expert demonstrations for behavior cloning.

Outputs:
- JSONL step logs (one file per player role)
- JSONL episode summaries (one file per format)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pokemon_rl.constants import FORMAT_LONG_HORIZON, FORMAT_MAIN
from pokemon_rl.env import LoggedSimpleHeuristicsPlayer, LoggingConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-battles", type=int, default=50)
    p.add_argument("--max-concurrent-battles", type=int, default=1)
    p.add_argument(
        "--formats",
        type=str,
        default=f"{FORMAT_MAIN},{FORMAT_LONG_HORIZON}",
        help="Comma-separated Showdown format IDs",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "data" / "demos",
        help="Output directory (gitignored by default)",
    )
    return p.parse_args()


async def _run_one_format(
    *,
    battle_format: str,
    n_battles: int,
    max_concurrent_battles: int,
    out_dir: Path,
) -> None:
    run_id = f"{battle_format}-{int(time.time())}"
    fmt_dir = out_dir / battle_format / run_id
    fmt_dir.mkdir(parents=True, exist_ok=True)

    # Avoid `|nametaken|` on repeated runs:
    # poke-env auto-generates usernames with a per-process counter, which resets
    # when you restart Python. Using rand=True makes the username unique.
    expert_account = AccountConfiguration.generate(f"LoggedSimpleHeur-{battle_format}", rand=True)
    opp_account = AccountConfiguration.generate(f"RandomPlayer-{battle_format}", rand=True)

    expert = LoggedSimpleHeuristicsPlayer(
        battle_format=battle_format,
        max_concurrent_battles=max_concurrent_battles,
        account_configuration=expert_account,
        logging_cfg=LoggingConfig(path=fmt_dir / "expert_steps.jsonl", run_id=run_id, role="expert"),
    )
    opponent = RandomPlayer(
        battle_format=battle_format,
        max_concurrent_battles=max_concurrent_battles,
        account_configuration=opp_account,
    )

    await expert.battle_against(opponent, n_battles=n_battles)

    # Episode summaries
    episodes_path = fmt_dir / "episodes.jsonl"
    for _tag, battle in expert.battles.items():
        # Only finished battles should exist here, but we keep it defensive.
        if getattr(battle, "finished", False):
            from pokemon_rl.logging import JsonlLogger

            JsonlLogger(episodes_path).log(expert.episode_summary(battle))

    win_rate = expert.n_won_battles / max(1, expert.n_finished_battles)
    print(
        f"[{battle_format}] expert vs random: "
        f"{expert.n_won_battles}/{expert.n_finished_battles} wins "
        f"(win_rate={win_rate:.3f})"
    )
    print(f"Wrote logs to: {fmt_dir}")


def main() -> None:
    args = _parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    async def runner() -> None:
        for fmt in formats:
            await _run_one_format(
                battle_format=fmt,
                n_battles=args.n_battles,
                max_concurrent_battles=args.max_concurrent_battles,
                out_dir=args.out_dir,
            )

    asyncio.run(runner())


if __name__ == "__main__":
    main()
