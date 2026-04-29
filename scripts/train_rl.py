#!/usr/bin/env python3
"""Train a PPO agent on Pokémon battles.

Supports two modes:
  1. From scratch  – ``--bc-checkpoint`` is not provided.
  2. BC warmstart  – pass ``--bc-checkpoint path/to/bc_best.pt`` to
     initialise the actor-critic trunk from a BC-pretrained policy.

Usage
-----
    # Mock environment (no Pokémon Showdown server needed):
    python scripts/train_rl.py \\
        --mock \\
        --total-timesteps 50000 \\
        --run-name scratch_run

    # BC warmstart with mock environment:
    python scripts/train_rl.py \\
        --mock \\
        --bc-checkpoint checkpoints/bc_best.pt \\
        --total-timesteps 50000 \\
        --run-name bc_warmstart_run

    # Live environment (requires server):
    python scripts/train_rl.py \\
        --total-timesteps 500000 \\
        --run-name scratch_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from src.agents.ppo_agent import PPOConfig
from src.environment.battle_env import MockBattleEnv
from src.training.rl_trainer import RLConfig, RLTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent for Pokémon battles.")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Use MockBattleEnv (no server required).")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC checkpoint for warm-start.")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default="rl_run")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--output-plot", type=str, default=None)
    args = parser.parse_args()

    ppo_cfg = PPOConfig(
        hidden_sizes=args.hidden_sizes,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )

    rl_cfg = RLConfig(
        ppo=ppo_cfg,
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        run_name=args.run_name,
        bc_checkpoint=args.bc_checkpoint,
    )

    if args.mock:
        print("Using MockBattleEnv (no Pokémon Showdown server required).")
        env_fn = lambda: MockBattleEnv(max_steps=50)
    else:
        print("Using live PokemonBattleEnv.")
        env_fn = _make_live_env(args)

    print(
        f"Starting PPO training: {args.total_timesteps} steps | "
        f"BC warmstart: {args.bc_checkpoint or 'none'}"
    )

    trainer = RLTrainer(env_fn=env_fn, config=rl_cfg)
    history = trainer.train()

    # Save history
    os.makedirs(args.save_dir, exist_ok=True)
    history_path = os.path.join(args.save_dir, f"{args.run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history → {history_path}")

    if args.output_plot and history:
        _plot_history(history, args.output_plot)


def _make_live_env(args):  # pragma: no cover
    """Return a factory for the live PokémonBattleEnv."""
    def env_fn():
        try:
            from poke_env.player import RandomPlayer
        except ImportError:
            print("poke-env is not installed.  Run: pip install poke-env")
            sys.exit(1)

        from src.environment.battle_env import PokemonBattleEnv
        # Placeholder – real implementation requires a running server
        raise NotImplementedError(
            "Live environment not yet configured.  Use --mock for testing."
        )
    return env_fn


def _plot_history(history, path: str) -> None:
    steps = [h["total_steps"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, [h["mean_ep_reward"] for h in history])
    axes[0].fill_between(
        steps,
        [h["mean_ep_reward"] - h["std_ep_reward"] for h in history],
        [h["mean_ep_reward"] + h["std_ep_reward"] for h in history],
        alpha=0.3,
    )
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Mean episode reward")
    axes[0].set_title("Episode Reward")

    axes[1].plot(steps, [h["win_rate"] for h in history])
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Win rate")
    axes[1].set_title("Win Rate")
    axes[1].set_ylim(0, 1)

    axes[2].plot(steps, [h["policy_loss"] for h in history], label="policy")
    axes[2].plot(steps, [h["value_loss"] for h in history], label="value")
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("PPO Losses")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved plot → {path}")


if __name__ == "__main__":
    main()
