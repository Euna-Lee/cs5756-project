#!/usr/bin/env python3
"""Evaluate a trained policy and produce comparison plots.

Supports:
  - Evaluating a single checkpoint against a mock environment.
  - Comparing a BC-initialised agent vs a from-scratch agent.
  - Generating win-rate vs sample-efficiency curves.

Usage
-----
    # Evaluate a single checkpoint:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/rl_run_final.pt \\
        --n-episodes 200 \\
        --mock

    # Compare two checkpoints (e.g. scratch vs BC warmstart):
    python scripts/evaluate.py \\
        --checkpoint checkpoints/scratch_final.pt \\
        --compare-checkpoint checkpoints/bc_warmstart_final.pt \\
        --label "Scratch" \\
        --compare-label "BC Warmstart" \\
        --history checkpoints/scratch_run_history.json \\
        --compare-history checkpoints/bc_warmstart_run_history.json \\
        --output-plot results/comparison.png \\
        --mock
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.environment.battle_env import MockBattleEnv
from src.environment.action_space import get_action_mask


def evaluate_agent(
    agent: PPOAgent,
    env_fn,
    n_episodes: int = 200,
) -> dict:
    """Run ``n_episodes`` evaluation episodes and return statistics."""
    env = env_fn()
    rewards = []
    lengths = []
    wins = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            mask = _get_mask(env)
            action, _, _ = agent.act(obs, action_mask=mask, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

        rewards.append(ep_reward)
        lengths.append(ep_length)
        wins.append(ep_reward > 0)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "win_rate": float(np.mean(wins)),
        "n_episodes": n_episodes,
    }


def _get_mask(env):
    battle = getattr(env, "_current_battle", None)
    if battle is not None:
        return np.array(get_action_mask(battle), dtype=bool)
    return None


def load_agent(checkpoint_path: str, device: str = "cpu") -> PPOAgent:
    agent = PPOAgent(PPOConfig(device=device))
    agent.load(checkpoint_path)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pokémon RL agents.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--compare-checkpoint", type=str, default=None)
    parser.add_argument("--label", type=str, default="Agent")
    parser.add_argument("--compare-label", type=str, default="Compare Agent")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--history", type=str, default=None,
                        help="JSON training history for the main checkpoint.")
    parser.add_argument("--compare-history", type=str, default=None,
                        help="JSON training history for the compare checkpoint.")
    parser.add_argument("--mock", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-plot", type=str, default=None)
    args = parser.parse_args()

    if args.mock:
        env_fn = lambda: MockBattleEnv(max_steps=50)
    else:
        raise NotImplementedError("Live evaluation requires a running server.  Use --mock.")

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = load_agent(args.checkpoint, device=args.device)
    stats = evaluate_agent(agent, env_fn, n_episodes=args.n_episodes)
    print(f"\n{'='*40}")
    print(f"  {args.label}")
    print(f"{'='*40}")
    print(f"  Win rate:      {stats['win_rate']:.2%}")
    print(f"  Mean reward:   {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean ep len:   {stats['mean_length']:.1f}")

    compare_stats = None
    compare_agent = None
    if args.compare_checkpoint:
        print(f"\nLoading compare checkpoint: {args.compare_checkpoint}")
        compare_agent = load_agent(args.compare_checkpoint, device=args.device)
        compare_stats = evaluate_agent(compare_agent, env_fn, n_episodes=args.n_episodes)
        print(f"\n{'='*40}")
        print(f"  {args.compare_label}")
        print(f"{'='*40}")
        print(f"  Win rate:      {compare_stats['win_rate']:.2%}")
        print(f"  Mean reward:   {compare_stats['mean_reward']:.2f} ± {compare_stats['std_reward']:.2f}")
        print(f"  Mean ep len:   {compare_stats['mean_length']:.1f}")

    if args.output_plot:
        _plot_comparison(
            stats, compare_stats,
            args.label, args.compare_label,
            args.history, args.compare_history,
            args.output_plot,
        )


def _plot_comparison(
    stats1, stats2,
    label1, label2,
    history_path1, history_path2,
    output_path,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    has_history = history_path1 is not None or history_path2 is not None

    n_panels = 3 if has_history else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel 0: Bar chart of final win rates
    labels = [label1]
    win_rates = [stats1["win_rate"]]
    colors = ["#4C72B0"]

    if stats2 is not None:
        labels.append(label2)
        win_rates.append(stats2["win_rate"])
        colors.append("#DD8452")

    axes[0].bar(labels, win_rates, color=colors, edgecolor="black", width=0.4)
    axes[0].set_ylabel("Win Rate")
    axes[0].set_title("Final Win Rate Comparison")
    axes[0].set_ylim(0, 1.0)
    for i, wr in enumerate(win_rates):
        axes[0].text(i, wr + 0.02, f"{wr:.1%}", ha="center", fontsize=11)

    if has_history and len(axes) > 1:
        # Panel 1: Win rate vs timesteps
        for hist_path, label, color in [
            (history_path1, label1, "#4C72B0"),
            (history_path2, label2, "#DD8452"),
        ]:
            if hist_path and os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)
                steps = [h["total_steps"] for h in history]
                wr = [h["win_rate"] for h in history]
                axes[1].plot(steps, wr, label=label, color=color)

        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Win Rate")
        axes[1].set_title("Win Rate vs Sample Efficiency")
        axes[1].legend()
        axes[1].set_ylim(0, 1.0)

        # Panel 2: Mean reward vs timesteps
        for hist_path, label, color in [
            (history_path1, label1, "#4C72B0"),
            (history_path2, label2, "#DD8452"),
        ]:
            if hist_path and os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)
                steps = [h["total_steps"] for h in history]
                rew = [h["mean_ep_reward"] for h in history]
                axes[2].plot(steps, rew, label=label, color=color)

        axes[2].set_xlabel("Environment Steps")
        axes[2].set_ylabel("Mean Episode Reward")
        axes[2].set_title("Sample Efficiency (Episode Reward)")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot → {output_path}")


if __name__ == "__main__":
    main()
