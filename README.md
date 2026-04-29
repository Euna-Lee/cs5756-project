# cs5756-project – Playing Pokémon with RL + Behaviour Cloning

**Team:** Luke Tao (lyt5), Ethan Zhang (epz6), Euna Lee (ekl49)

## Project Description

We train a reinforcement-learning agent to play competitive Pokémon battles and
test whether **behaviour cloning (BC) used as a warm-start** improves sample
efficiency and early-stage performance compared to training from scratch.

Specifically we:
1. Collect expert demonstrations from a strong heuristic bot (`SimpleHeuristicsPlayer`).
2. Pre-train a neural policy via behaviour cloning (cross-entropy imitation loss).
3. Fine-tune with PPO reinforcement learning.
4. Evaluate whether BC-initialised agents reach a target win rate faster and with
   fewer environment interactions than purely RL-trained baselines.

The platform is [**poke-env**](https://github.com/hsahovic/poke-env), a Python
environment that interfaces with Pokémon Showdown, providing programmatic control
of battles for self-play or fixed-opponent training.

---

## Repository Structure

```
cs5756-project/
├── requirements.txt
├── src/
│   ├── environment/
│   │   ├── obs_encoder.py      # Battle state → 384-dim numpy array
│   │   ├── action_space.py     # 9-action space (4 moves + 5 switches)
│   │   └── battle_env.py       # gymnasium.Env wrapper + MockBattleEnv
│   ├── agents/
│   │   ├── expert_bot.py       # MaxDamagePlayer & SimpleHeuristicsPlayer
│   │   ├── policy_network.py   # Shared-trunk actor-critic MLP (PyTorch)
│   │   └── ppo_agent.py        # PPO agent with GAE rollout buffer
│   ├── training/
│   │   ├── bc_trainer.py       # Behaviour cloning training loop + metrics
│   │   └── rl_trainer.py       # PPO training loop (scratch or BC warmstart)
│   └── utils/
│       └── battle_logger.py    # JSONL logger for observations/actions/rewards
├── scripts/
│   ├── collect_demonstrations.py
│   ├── train_bc.py
│   ├── train_rl.py
│   └── evaluate.py
└── tests/                      # 93 unit tests (no server required)
```

---

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** Running against a live Pokémon Showdown server additionally requires
> `poke-env >= 0.7.1` and a locally running server.  All scripts support a
> `--mock` flag that skips the server and uses a lightweight synthetic environment
> for development and testing.

---

## Quick Start (mock mode – no server required)

### 1. Collect expert demonstrations

```bash
python scripts/collect_demonstrations.py \
    --mock \
    --n-battles 1000 \
    --output-path data/expert_demos.jsonl
```

### 2. Train behaviour cloning policy

```bash
python scripts/train_bc.py \
    --data data/expert_demos.jsonl \
    --save-dir checkpoints \
    --n-epochs 50 \
    --output-plot results/bc_curves.png
```

### 3. Train RL agents

```bash
# From scratch baseline
python scripts/train_rl.py \
    --mock \
    --total-timesteps 500000 \
    --run-name scratch \
    --save-dir checkpoints \
    --output-plot results/scratch_curves.png

# BC warm-start
python scripts/train_rl.py \
    --mock \
    --bc-checkpoint checkpoints/bc_best.pt \
    --total-timesteps 500000 \
    --run-name bc_warmstart \
    --save-dir checkpoints \
    --output-plot results/bc_warmstart_curves.png
```

### 4. Evaluate and compare

```bash
python scripts/evaluate.py \
    --mock \
    --checkpoint checkpoints/scratch_final.pt \
    --compare-checkpoint checkpoints/bc_warmstart_final.pt \
    --label "Scratch" \
    --compare-label "BC Warmstart" \
    --history checkpoints/scratch_history.json \
    --compare-history checkpoints/bc_warmstart_history.json \
    --output-plot results/comparison.png
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 93 tests run offline (no Pokémon Showdown server needed).

---

## Observation Space (384 floats)

| Slice        | Description                                              |
|--------------|----------------------------------------------------------|
| `[0:192]`    | **Player side** – active Pokémon (51) + moves (96) + bench (45) |
| `[192:384]`  | **Opponent side** – same layout as player side           |

**Active Pokémon (51 dims):** HP fraction · status one-hot (7) · type-1 one-hot (18) · type-2 one-hot+none (19) · normalised base stats (6).

**Move (24 dims):** type one-hot (18) · normalised base power · accuracy · category one-hot (3) · PP fraction.

**Bench slot (9 dims):** HP fraction · status one-hot (7) · alive flag.

## Action Space (9 actions)

| Index | Meaning                     |
|-------|-----------------------------|
| 0–3   | Use move slot 0–3           |
| 4–8   | Switch to bench slot 0–4    |

Invalid actions are masked to −∞ before the softmax so the policy never
selects an unavailable action.

---

## Algorithm: PPO with BC Warm-start

```
Expert demos  ──► BC training ──► BC policy θ_BC
                                        │
                         ┌──────────────┘
                         ▼
Random init θ_0       θ_BC
      │                   │
      ▼                   ▼
   PPO (scratch)    PPO (warm-start)
      │                   │
      ▼                   ▼
  π_scratch(s)       π_warm(s)
```

**Reward shaping:**

| Event                     | Reward  |
|---------------------------|---------|
| Win battle                | +30     |
| Lose battle               | −30     |
| Opponent Pokémon faints   | +2      |
| Player Pokémon faints     | −2      |
| Opponent HP dealt / lost  | ±1 × Δhp |

---

## Team Roles

| Member | Responsibilities |
|--------|-----------------|
| **Ethan Zhang** | PPO implementation (`ppo_agent.py`, `rl_trainer.py`), reward function, RL experiments |
| **Luke Tao**    | Observation/action space (`obs_encoder.py`, `action_space.py`), BC training loop (`bc_trainer.py`) |
| **Euna Lee**    | Environment setup (`battle_env.py`), logging (`battle_logger.py`), data-collection scripts |
