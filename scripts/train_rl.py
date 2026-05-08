#!/usr/bin/env python3
"""On-policy RL fine-tuning (PPO-style) vs a fixed opponent."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pokemon_rl.models import MaskedPolicyNet
from pokemon_rl.representations import ACTION_SPACE_SIZE
from pokemon_rl.rewards import terminal_reward
from pokemon_rl.rl.on_policy_player import OnPolicyPolicyPlayer

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", type=str, required=True)
    p.add_argument("--runs-dir", type=Path, default=_REPO_ROOT / "runs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)

    p.add_argument("--init-checkpoint", type=Path, default=None, help="Optional BC checkpoint (.pt)")
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--battles-per-update", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)

    # PPO-ish knobs
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--clip-eps", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Determine obs_dim from our tabular encoder (fixed)
    obs_dim = 13

    policy = MaskedPolicyNet(obs_dim=obs_dim, hidden=args.hidden).to(args.device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        policy.load_state_dict(ckpt["model_state_dict"])

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    run_id = f"rl-{args.format}-{'bc' if args.init_checkpoint else 'scratch'}-{int(time.time())}"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    async def train_loop():
        agent = OnPolicyPolicyPlayer(
            policy=policy,
            device=args.device,
            battle_format=args.format,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate(f"RLAgent-{args.format}", rand=True),
        )
        opp = RandomPlayer(
            battle_format=args.format,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate(f"RLOpp-{args.format}", rand=True),
        )

        global_step = 0
        for update in range(args.updates):
            # Collect episodes
            await agent.battle_against(opp, n_battles=args.battles_per_update)
            episodes = agent.drain_finished()

            # Build one big batch of (x, mask, action, old_logp, return)
            xs = []
            masks = []
            actions = []
            old_logps = []
            returns = []
            wins = 0
            n_eps = 0

            for ep in episodes:
                n_eps += 1
                if ep.won:
                    wins += 1
                R = float(ep.terminal_reward)
                for t in ep.transitions:
                    xs.append(t.x)
                    masks.append(t.mask)
                    actions.append(t.action)
                    old_logps.append(t.logp)
                    returns.append(R)

            if not xs:
                print("No transitions collected; skipping update")
                continue

            X = torch.stack(xs).to(args.device)
            M = torch.stack(masks).to(args.device)
            A = torch.tensor(actions, dtype=torch.long, device=args.device)
            old_logp = torch.tensor(old_logps, dtype=torch.float32, device=args.device)
            R = torch.tensor(returns, dtype=torch.float32, device=args.device)

            # Advantage: simple baseline
            adv = R - R.mean()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            policy.train()
            for _ in range(args.ppo_epochs):
                logits = policy.mask_logits(policy(X).logits, M)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(A)
                ratio = torch.exp(new_logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                loss = -(torch.min(ratio * adv, clipped * adv)).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            win_rate = wins / max(1, n_eps)
            writer.add_scalar("train/win_rate", win_rate, update)
            writer.add_scalar("train/loss", float(loss.item()), update)
            global_step += len(xs)
            print(f"update {update+1}/{args.updates} win_rate={win_rate:.3f} steps={len(xs)}")

            torch.save(
                {
                    "format": args.format,
                    "obs_dim": obs_dim,
                    "action_space_size": ACTION_SPACE_SIZE,
                    "model_state_dict": policy.state_dict(),
                    "hidden": args.hidden,
                    "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                },
                run_dir / f"policy_update{update+1}.pt",
            )

        writer.close()
        print(f"Saved RL checkpoints under: {run_dir}")

    import asyncio

    asyncio.run(train_loop())


if __name__ == "__main__":
    main()
