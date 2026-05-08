#!/usr/bin/env python3
"""Evaluate a checkpoint policy vs a fixed opponent."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import torch
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pokemon_rl.models import MaskedPolicyNet
from pokemon_rl.rl.on_policy_player import OnPolicyPolicyPlayer
from pokemon_rl.logging import JsonlLogger

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n-battles", type=int, default=100)
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "runs" / "eval.jsonl")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    obs_dim = int(ckpt.get("obs_dim", 13))
    hidden = int(ckpt.get("hidden", 128))
    policy = MaskedPolicyNet(obs_dim=obs_dim, hidden=hidden).to(args.device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    async def runner() -> None:
        agent = OnPolicyPolicyPlayer(
            policy=policy,
            device=args.device,
            battle_format=args.format,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate(f"EvalAgent-{args.format}", rand=True),
        )
        opp = RandomPlayer(
            battle_format=args.format,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate(f"EvalOpp-{args.format}", rand=True),
        )
        await agent.battle_against(opp, n_battles=args.n_battles)

        wins = 0
        finished = 0
        for _tag, battle in agent.battles.items():
            if getattr(battle, "finished", False):
                finished += 1
                if getattr(battle, "won", False):
                    wins += 1

        rec = {
            "kind": "eval",
            "ts": time.time(),
            "format": args.format,
            "checkpoint": str(args.checkpoint),
            "n_battles": args.n_battles,
            "finished": finished,
            "wins": wins,
            "win_rate": wins / max(1, finished),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        JsonlLogger(args.out).log(rec)
        print(rec)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
