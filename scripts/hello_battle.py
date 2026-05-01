#!/usr/bin/env python3
"""Smoke test: two RandomPlayer agents battle on your local Showdown instance."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from poke_env.player import RandomPlayer

from pokemon_rl.utils.config import load_yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    default_config = _repo_root() / "configs" / "default.yaml"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="YAML with battle_format, n_battles, max_concurrent_battles",
    )
    p.add_argument("--battle-format", type=str, default=None, help="Override config battle_format")
    p.add_argument("--n-battles", type=int, default=None, help="Override config n_battles")
    p.add_argument(
        "--max-concurrent-battles",
        type=int,
        default=None,
        help="Override config max_concurrent_battles",
    )
    return p.parse_args()


async def _main_async(
    *,
    battle_format: str,
    n_battles: int,
    max_concurrent_battles: int,
) -> None:
    p1 = RandomPlayer(battle_format=battle_format, max_concurrent_battles=max_concurrent_battles)
    p2 = RandomPlayer(battle_format=battle_format, max_concurrent_battles=max_concurrent_battles)
    await p1.battle_against(p2, n_battles=n_battles)
    print(f"Finished battles: {p1.n_finished_battles}")
    print(f"Player 1 wins: {p1.n_won_battles}")


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config) if args.config.is_file() else {}
    battle_format = args.battle_format or str(cfg.get("battle_format", "gen9bssfactory"))
    n_battles = args.n_battles if args.n_battles is not None else int(cfg.get("n_battles", 1))
    max_cb = (
        args.max_concurrent_battles
        if args.max_concurrent_battles is not None
        else int(cfg.get("max_concurrent_battles", 1))
    )
    try:
        asyncio.run(
            _main_async(
                battle_format=battle_format,
                n_battles=n_battles,
                max_concurrent_battles=max_cb,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
