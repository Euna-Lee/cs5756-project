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

## Battle format IDs and what they are **not**

| ID | What it is (Showdown) | Common misuse in write-ups |
|----|------------------------|----------------------------|
| `gen9bssfactory` | Official description: **Randomized 3v3 Singles** — factory/BSS-flavored sets, team preview, **three** active Pokémon per game (Flat Rules, VGC timer). | Calling it generic “3v3” without noting **preview + BSS rules** overstates similarity to VGC doubles or other 3v3 rulesets. |
| `gen9randombattle` | Standard **Gen 9 Random Battle**: six Pokémon, **sequential singles**, globally generated “competitively viable” sets. | Equating this to **full competitive 6v6 ladder** (OU, custom teams, different clauses) is imprecise; use it as a **long-horizon / full-roster-size** proxy, not a ladder simulator. |
| `gen9battlefactory` | **Gen 9 Battle Factory**: **6v6** singles, auto-generated teams drawn from **tier-themed** factory pools (still no human teambuilder). | Optional **pilot** format: closer to “six used in order under tier-shaped random teams” than `gen9randombattle`, but still not OU/Ubers with hand-built teams. |

**Bottom line:** `gen9bssfactory` vs `gen9randombattle` is a clean **three-actives + preview** vs **six sequential randbats** contrast. It does **not** map one-to-one to “VGC 3v3 vs OU 6v6”; state that limitation in the report.

### Pilot: `gen9battlefactory` (1–2 seeds)

Collect demos, then run the same Week 3 pipeline on this format only (cheap sanity check):

```bash
python scripts/collect_demos.py --formats gen9battlefactory --n-battles 30
python scripts/week3_sweep.py --seeds 0,1 --formats gen9battlefactory --rl-updates 30 --eval-every 10 --execute --device cuda
```

Use `plot_week3.py` on the resulting `eval.jsonl` as usual. Compare trends qualitatively (small *n*): same code path, different format generator and horizon.

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
- `gen9bssfactory` (Showdown: **Randomized 3v3 Singles** — preview, three actives)
- `gen9randombattle` (six Pokémon, sequential singles randbats)

To add the optional **6v6 Battle Factory** pilot format: `--formats gen9bssfactory,gen9randombattle,gen9battlefactory` (collect demos for each format you train on).

It writes JSONL logs under `data/demos/<format>/<run_id>/`:
- `expert_steps.jsonl`: per-decision records (obs snapshot + legal mask + action id when available)
- `episodes.jsonl`: per-battle summaries including terminal reward and win/loss

## Week 2: BC + RL

Train on demos collected under `data/demos/<format>/*/expert_steps.jsonl`. For **RL** and **eval**, keep local Showdown running (same as Week 1).

Each training run writes **`run_manifest.json`** in its run directory (`git` commit, `argv`, hyperparameters, demo paths or RL settings). Checkpoints include a nested **`run_metadata`** field for traceability.

### Behavior cloning (BC)

Trains a masked policy on expert steps; writes checkpoints and TensorBoard under `runs/bc-<format>-<timestamp>/`.

The first file **`policy_epoch0.pt`** is the **untrained** network (before any gradient step). Use it as a “structured random init” baseline; for a **uniform random legal-action** baseline in eval, use `eval.py --random-policy` (see below).

```bash
python scripts/train_bc.py --format gen9bssfactory
python scripts/train_bc.py --format gen9randombattle
```

Useful options: `--epochs`, `--batch-size`, `--lr`, `--val-frac`, `--hidden`, `--device`, `--demos-root`, `--runs-dir`, `--run-name`, `--force`.

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

Useful options: **`--updates`**, **`--battles-per-update`**, **`--lr`**, **`--ppo-epochs`**, **`--clip-eps`**, **`--hidden`**, **`--seed`**, **`--device`**, **`--runs-dir`**, **`--opponent`** (`random` = `RandomPlayer`, `heuristic` = `SimpleHeuristicsPlayer`), **`--run-name`**, **`--force`**.

**`policy_update0.pt`** is saved **before** any RL collection/update (scratch initialization or BC weights only).

### Evaluation + metrics

Evaluate a **trained checkpoint** vs an opponent (`--opponent random` = `RandomPlayer`, or `heuristic` = `SimpleHeuristicsPlayer`). Each run appends one JSON object to `--out` (includes `seed`, opponent, `run_metadata` with `git_commit`).

```bash
python scripts/eval.py --format gen9bssfactory --checkpoint runs/<bc-run>/policy_epoch5.pt --n-battles 200 --seed 0 --opponent random --out runs/eval.jsonl
```

**Uniform random legal actions** (no neural net — good sanity check vs ~50% win rate):

```bash
python scripts/eval.py --format gen9bssfactory --random-policy --n-battles 200 --seed 0 --out runs/eval.jsonl
```

Turn eval logs into a CSV for plots / reports:

```bash
python scripts/plot_winrate.py --eval-jsonl runs/eval.jsonl --out-csv runs/eval.csv
```

For hypothesis comparisons, train/eval **separately per format** (e.g. `gen9bssfactory` vs `gen9randombattle`, optionally `gen9battlefactory` as a stricter 6v6 factory pilot) and compare RL-from-scratch vs RL-from-BC on the same opponent and eval budget.

### Week 2 exit checks (before Week 3 compute)

1. **BC beats baselines**  
   - Eval `--random-policy` and `policy_epoch0.pt` (or epoch 1) vs the same `--opponent` and `--n-battles` (e.g. 200).  
   - Eval a **trained** BC checkpoint (`policy_epoch3`–`policy_epoch5`).  
   - Expect BC **noticeably above ~50%** win rate vs `RandomPlayer` if labels, masks, and obs are consistent.

2. **RL warm-start vs scratch (small A/B)**  
   Use the **same** `--format`, `--seed`, `--updates`, `--battles-per-update`, and eval schedule. Example (20 updates, then eval checkpoints at 0 / 5 / 10 / 20):

   ```bash
   SEED=0
   python scripts/train_rl.py --format gen9bssfactory --seed $SEED --updates 20 --battles-per-update 20
   python scripts/train_rl.py --format gen9bssfactory --seed $SEED --updates 20 --battles-per-update 20 \
     --init-checkpoint runs/<bc-run>/policy_epoch5.pt
   ```

   For each run, evaluate:

   ```bash
   for U in 0 5 10 20; do
     python scripts/eval.py --format gen9bssfactory --checkpoint runs/<rl-run>/policy_update${U}.pt \
       --n-battles 200 --seed $SEED --opponent random --out runs/eval.jsonl
   done
   ```

   Early curve should favor warm-start **if** the BC checkpoint loads and masking matches; if not, inspect `init_checkpoint` path and `run_manifest.json`.

3. **Reproducibility**  
   Archive **`run_manifest.json`**, eval lines (with `run_metadata`), and note **git commit** before large Week 3 sweeps.

## Week 3: main comparisons (multi-seed, per format)

This week runs **BC-only**, **RL from scratch**, and **RL warm-start from BC** with matched seeds and hyperparameters, then aggregates evals for plots.

### Design choices (implemented defaults)

- **Orchestrator** `scripts/week3_sweep.py` **prints commands by default** (`PLAN.txt` + `sweep_manifest.json`). Use **`--execute`** only when you intend to spend compute (Showdown + GPU time).
- **Default seeds:** `0,1,2` (override with `--seeds 0,1,2,3,4` for 5 seeds).
- **Default formats:** `gen9bssfactory,gen9randombattle`. Optional **pilot:** `gen9battlefactory` (6v6 factory — see [Battle format IDs](#battle-format-ids-and-what-they-are-not)).
- **RL training opponent:** `--train-opponent random` (easy) or `heuristic` (harder). Independent of eval opponent.
- **Eval opponent:** `--eval-opponent random` (curve sanity) or `heuristic` (stress test). Same eval settings for all three conditions.
- **Deterministic run names:** `w3-<sweep_id>-bc-<format>-s<seed>`, `...-rl-scratch-...`, `...-rl-warm-...` under `runs/`.

### One-command sweep (dry-run first)

```bash
# Preview the full command list (no training)
python scripts/week3_sweep.py

# Smaller smoke sweep (1 seed, 1 format, fewer RL steps)
python scripts/week3_sweep.py --seeds 0 --formats gen9bssfactory --rl-updates 20 --eval-every 10

# Run for real (long): uses local Showdown for every train + eval step
python scripts/week3_sweep.py --execute --device cuda
```

Outputs per sweep id:

- `runs/week3_sweep_<id>/PLAN.txt` — all shell commands
- `runs/week3_sweep_<id>/sweep_manifest.json` — hyperparameters + eval schedule
- `runs/week3_sweep_<id>/eval.jsonl` — all eval rows with `experiment` metadata (`condition`, `seed`, `rl_update`, `battles_per_update`, `rl_train_battles`, …). After evaluation, `experiment` may also include `cumulative_train_battles` / `cumulative_train_env_steps` copied from RL checkpoints when present.

### Manual pieces (same as orchestrator)

- **BC:** `train_bc.py --run-name <stable-name> ...`
- **RL:** `train_rl.py --opponent random|heuristic --run-name ...` and warm-start with `--init-checkpoint`
- **Eval:** `eval.py --extra-json '<json>'` for structured Week 3 tags (the sweep does this for you)

### Plot comparison curves

**Sample-efficiency axis (for the report):** “RL update” alone is a weak unit if `battles_per_update` or code paths ever differ. Prefer **`--x-axis rl_train_battles`** (default): cumulative **RL training** battles completed before the evaluated checkpoint (`rl_update × battles_per_update` from the sweep, or exact counts from the checkpoint after re-eval with current `eval.py`). For finer-grained credit assignment, use **`--x-axis rl_train_env_steps`** (requires RL checkpoints saved by the current `train_rl.py`, which logs transition counts).

The horizontal **BC-only** line is **offline** supervised learning on demos — it does **not** consume RL training battles or env steps on that axis, so do not treat it as a point on the RL interaction curve; use it only as a performance reference.

After `eval.jsonl` has enough rows:

```bash
python scripts/plot_week3.py --eval-jsonl runs/week3_sweep_<id>/eval.jsonl --sweep-id <id> --out runs/week3_sweep_<id>/curves.png
# Same data vs PPO update index (legacy-style x-axis):
python scripts/plot_week3.py --eval-jsonl runs/week3_sweep_<id>/eval.jsonl --sweep-id <id> --x-axis rl_update --out runs/week3_sweep_<id>/curves_by_update.png
# Tabular summary for the write-up:
python scripts/plot_week3.py --eval-jsonl runs/week3_sweep_<id>/eval.jsonl --sweep-id <id> --dump-curve-csv runs/week3_sweep_<id>/curve_summary.csv
```

This plots **mean ± std over seeds** for `rl_scratch` and `rl_warm`, and a horizontal **BC-only** line per format.

By default the plot includes only eval rows with **`opponent` random** (legacy rows without that field count as random). After appending heuristic re-evals to the same file, pass **`--eval-opponent heuristic`** for the harder curve, or **`--eval-opponent any`** only if the file is not mixed.

Use the **`week3_sweep_<id>`** folder whose `<id>` matches your `w3-<id>-bc-...` run directories (e.g. `1778448153`). A folder that only has `PLAN.txt` and `sweep_manifest.json` was a **dry run** (`--execute` not used) — there is no `eval.jsonl` there.

After `--execute`, you should see `runs/week3_sweep_<id>/eval.jsonl`. The first line may be `sweep_meta` (sweep started); **`kind: eval` lines appear only after each `eval.py` finishes** (200 battles can take a while). If the file is missing entirely, the sweep likely **stopped at the first error** (e.g. `train_bc` refusing an existing `--run-name` without `--force`): check the terminal for `CalledProcessError` or `SystemExit`.

### Parameter / difficulty variations (Week 3 scope)

- **Formats:** primary pair `gen9bssfactory` vs `gen9randombattle`; optional `gen9battlefactory` pilot (1–2 seeds) for a different 6v6 random regime.
- **Opponent difficulty:** rerun a sweep with `--train-opponent heuristic` and/or `--eval-opponent heuristic`.
- **RL depth:** `--rl-updates`, `--battles-per-update`, `--eval-every`.

### Sanity check: BC vs warm-start @ update 0

For a given `sweep_id`, BC-only win rate and RL-warm `policy_update0.pt` eval should agree within noise (same weights). If not, check checkpoint paths and duplicate rows in `eval.jsonl`:

```bash
python scripts/compare_bc_warm_zero.py --eval-jsonl runs/week3_sweep_<id>/eval.jsonl --sweep-id <id>
```

Optional: `--out-csv runs/week3_sweep_<id>/bc_warm0_check.csv`

### Heuristic re-eval (non-saturated curves)

If the sweep used random eval opponents and `gen9randombattle` saturates near 1.0, append heuristic evals for every unique checkpoint referenced in the sweep (deduped by path, format, seed):

```bash
python scripts/reval_heuristic.py --from-eval-jsonl runs/week3_sweep_<id>/eval.jsonl --sweep-id <id> --device cuda
# dry-run: add --dry-run
python scripts/plot_week3.py --eval-jsonl runs/week3_sweep_<id>/eval_heuristic.jsonl --sweep-id <id> --eval-opponent heuristic --out runs/week3_sweep_<id>/curves_heuristic.png
```

### If warm-start drifts down quickly

Try a smaller RL step on BC weights: `python scripts/train_rl.py ... --init-checkpoint ... --warm-lr 1e-4` (or tune `--ppo-epochs`). `run_manifest.json` records `optimizer_lr` vs `lr`.

## Team

Luke Tao (lyt5), Ethan Zhang (epz6), Euna Lee (ekl49)
