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

- **Main (3v3-style random)**: `gen9bssfactory` (bring 6, choose 3)
- **Long-horizon baseline (6v6)**: `gen9randombattle`

## Layout

| Path | Purpose |
|------|---------|
| `configs/` | YAML defaults (format, run settings) |
| `src/pokemon_rl/` | Library code (`agents`, `env`, `logging`, `utils`) |
| `scripts/` | Runnable entry points (`hello_battle.py`, training stubs) |
| `data/` | Datasets / demos (gitignored) |
| `runs/` | Checkpoints, TensorBoard (gitignored) |

## Logging (JSONL)

Use `pokemon_rl.logging.JsonlLogger` to append one JSON object per line, for example step records with `battle_id`, `turn`, `action`, `reward`, `done`, etc.

## Team

Luke Tao (lyt5), Ethan Zhang (epz6), Euna Lee (ekl49)
