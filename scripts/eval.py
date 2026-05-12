#!/usr/bin/env python3
"""Evaluate a policy vs a fixed opponent (checkpoint, or uniform-random over legal actions)."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import torch
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pokemon_rl.logging import JsonlLogger
from pokemon_rl.models import MaskedPolicyNet
from pokemon_rl.rl.on_policy_player import OnPolicyPolicyPlayer
from pokemon_rl.rl.random_legal_player import RandomLegalPlayer
from pokemon_rl.utils.run_metadata import common_run_metadata


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", type=str, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, default=None, help="Trained policy .pt (BC or RL)")
    g.add_argument(
        "--random-policy",
        action="store_true",
        help="Baseline: uniform random over legal actions (fixed 10-action space); ignores --checkpoint",
    )
    p.add_argument("--n-battles", type=int, default=100)
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "runs" / "eval.jsonl")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=("random", "heuristic"),
        help="Opponent type: RandomPlayer or SimpleHeuristicsPlayer",
    )
    p.add_argument("--eval-note", type=str, default="", help="Optional label stored in the eval record")
    p.add_argument(
        "--extra-json",
        type=str,
        default=None,
        help='Merge JSON object into the eval record under key "experiment" (e.g. Week 3 sweep metadata).',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy: MaskedPolicyNet | None = None
    obs_dim = 13
    hidden = 128
    ckpt_interaction: dict = {}
    if args.checkpoint and not args.random_policy:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        obs_dim = int(ckpt.get("obs_dim", 13))
        hidden = int(ckpt.get("hidden", 128))
        policy = MaskedPolicyNet(obs_dim=obs_dim, hidden=hidden).to(args.device)
        policy.load_state_dict(ckpt["model_state_dict"])
        policy.eval()
        for k in ("cumulative_train_battles", "cumulative_train_env_steps", "policy_update"):
            if k in ckpt and ckpt[k] is not None:
                ckpt_interaction[k] = ckpt[k]

    meta = common_run_metadata(
        "eval.py",
        _REPO_ROOT,
        seed=args.seed,
        extra={
            "format": args.format,
            "opponent": args.opponent,
            "n_battles": args.n_battles,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "random_policy": bool(args.random_policy),
            "device": args.device,
            "eval_note": args.eval_note or None,
        },
    )

    async def runner() -> None:
        if args.random_policy:
            agent = RandomLegalPlayer(
                battle_format=args.format,
                max_concurrent_battles=1,
                account_configuration=AccountConfiguration.generate(f"EvalRandomLegal-{args.format}", rand=True),
            )
        else:
            assert policy is not None
            agent = OnPolicyPolicyPlayer(
                policy=policy,
                device=args.device,
                battle_format=args.format,
                max_concurrent_battles=1,
                account_configuration=AccountConfiguration.generate(f"EvalAgent-{args.format}", rand=True),
            )

        if args.opponent == "random":
            opp = RandomPlayer(
                battle_format=args.format,
                max_concurrent_battles=1,
                account_configuration=AccountConfiguration.generate(f"EvalOpp-{args.format}", rand=True),
            )
        else:
            opp = SimpleHeuristicsPlayer(
                battle_format=args.format,
                max_concurrent_battles=1,
                account_configuration=AccountConfiguration.generate(f"EvalOppHeur-{args.format}", rand=True),
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
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "random_policy": bool(args.random_policy),
            "opponent": args.opponent,
            "seed": args.seed,
            "n_battles": args.n_battles,
            "finished": finished,
            "wins": wins,
            "win_rate": wins / max(1, finished),
            "run_metadata": meta,
        }
        exp: dict = {}
        if args.extra_json:
            exp.update(json.loads(args.extra_json))
        for k, v in ckpt_interaction.items():
            if k not in exp:
                exp[k] = v
        if exp:
            rec["experiment"] = exp
        args.out.parent.mkdir(parents=True, exist_ok=True)
        JsonlLogger(args.out).log(rec)
        print(rec)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
