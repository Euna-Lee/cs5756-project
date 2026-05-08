# cs5756-project

RL vs behavior-cloning warm start for competitive Pokémon battles using [poke-env](https://github.com/hsahovic/poke-env) and a local [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) server.

## Prerequisites

- Python **3.10+**
- Node.js (for Showdown — you already have this if the server runs)
- Local Showdown, typically:

  ```bash
  cd pokemon-showdown
  node pokemon-showdown start --no-security
  ```

  Use this only on trusted / lab machines.

## Install (this repo)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

This installs the project in **editable** mode (`pyproject.toml` distribution name `cs5756-pokemon-rl`) so you can `import pokemon_rl` from anywhere in the venv.

## Hello battle (connectivity check)

With Showdown running locally (default WebSocket: `ws://localhost:8000/showdown/websocket`):

```bash
python scripts/hello_battle.py
```

Defaults live in `configs/default.yaml` (`gen9bssfactory` = bring 6, battle with 3). Override, for example:

```bash
python scripts/hello_battle.py --battle-format gen9randombattle --n-battles 3
```

## Battle Format IDs

- Main (3v3-style random): `gen9bssfactory` (bring 6, choose 3)
- Long-horizon baseline (6v6): `gen9randombattle`

## Layout

| Path | Purpose |
|------|---------|
| `configs/` | YAML defaults (format, run settings) |
| `src/pokemon_rl/` | Library code (`agents`, `env`, `logging`, `utils`) |
| `scripts/` | Entry points: demos, `train_bc.py`, `train_rl.py`, `eval.py`, etc. |
| `data/` | Datasets / demos (gitignored) |
| `runs/` | Checkpoints, TensorBoard (gitignored) |

## Logging (JSONL)

Use `pokemon_rl.logging.JsonlLogger` to append one JSON object per line, for example step records with `battle_id`, `turn`, `action`, `reward`, `done`, etc.

## Collect demos + baseline expert win rate (Week 1)

With local Showdown running:

```bash
python scripts/collect_demos.py --n-battles 50
```

This runs `SimpleHeuristicsPlayer` (expert) vs `RandomPlayer` on both:
- `gen9bssfactory` (3v3-style: bring 6, choose 3)
- `gen9randombattle` (6v6)

It writes JSONL logs under `data/demos/<format>/<run_id>/`:
- `expert_steps.jsonl`: per-decision records (obs snapshot + legal mask + action id when available)
- `episodes.jsonl`: per-battle summaries including terminal reward and win/loss

## Week 2: BC + RL

Train on demos collected under `data/demos/<format>/*/expert_steps.jsonl`. For **RL** and **eval**, keep local Showdown running (same as Week 1).

### Behavior cloning (BC)

Trains a masked policy on expert steps; writes checkpoints and TensorBoard under `runs/bc-<format>-<timestamp>/`.

```bash
python scripts/train_bc.py --format gen9bssfactory
python scripts/train_bc.py --format gen9randombattle
```

Useful options: `--epochs`, `--batch-size`, `--lr`, `--val-frac`, `--hidden`, `--device`, `--demos-root`, `--runs-dir`.

View training curves:

```bash
tensorboard --logdir runs
```

### Reinforcement learning (RL)

On-policy fine-tuning vs a fixed `RandomPlayer`. Checkpoints go under `runs/rl-<format>-scratch-<timestamp>/` or `runs/rl-<format>-bc-<timestamp>/` when warm-starting.

**From scratch:**

```bash
python scripts/train_rl.py --format gen9bssfactory
python scripts/train_rl.py --format gen9randombattle
```

**Warm-start from a BC checkpoint:**

```bash
python scripts/train_rl.py --format gen9bssfactory --init-checkpoint runs/<bc-run>/policy_epoch5.pt
```

Useful options: `--updates`, `--battles-per-update`, `--lr`, `--ppo-epochs`, `--clip-eps`, `--hidden`, `--seed`, `--device`, `--runs-dir`.

### Evaluation + metrics

Run **N** battles vs `RandomPlayer` and append one JSON line per run to `runs/eval.jsonl` (or another path via `--out`):

```bash
python scripts/eval.py --format gen9bssfactory --checkpoint runs/<run>/policy_update50.pt --n-battles 100 --out runs/eval.jsonl
```

Turn eval logs into a CSV for plots / reports:

```bash
python scripts/plot_winrate.py --eval-jsonl runs/eval.jsonl --out-csv runs/eval.csv
```

For hypothesis comparisons, train/eval **separately per format** (`gen9bssfactory` vs `gen9randombattle`) and compare RL-from-scratch vs RL-from-BC on the same opponent and eval budget.

## Team

Luke Tao (lyt5), Ethan Zhang (epz6), Euna Lee (ekl49)
