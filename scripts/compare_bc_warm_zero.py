#!/usr/bin/env python3
"""
Sanity-check warm-start: compare BC-only eval vs RL-warm at update 0 (same seed/format).

Large |delta| often means wrong checkpoint, obs mismatch, or mixed sweep rows in eval.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--sweep-id", type=int, required=True)
    p.add_argument("--format", type=str, default=None, help="Optional: single format filter")
    p.add_argument("--warn-delta", type=float, default=0.08, help="Print warning if |bc - warm0| exceeds this")
    p.add_argument(
        "--eval-opponent",
        type=str,
        default="random",
        choices=("random", "heuristic", "any"),
        help="Match plot_week3: only rows with this eval opponent (legacy rows without `opponent` count as random).",
    )
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV path")
    return p.parse_args()


def _load_rows(path: Path, sweep_id: int, *, eval_opponent: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "eval":
                continue
            exp = rec.get("experiment")
            if not isinstance(exp, dict) or exp.get("sweep_id") != sweep_id:
                continue
            top_opp = rec.get("opponent")
            if eval_opponent == "random":
                if top_opp == "heuristic":
                    continue
            elif eval_opponent == "heuristic":
                if top_opp != "heuristic":
                    continue
            rows.append(rec)
    return rows


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.eval_jsonl, args.sweep_id, eval_opponent=args.eval_opponent)
    if not rows:
        raise SystemExit(f"No matching eval rows for sweep_id={args.sweep_id} in {args.eval_jsonl}")

    # (format, seed) -> bc win_rate
    bc: dict[tuple[str, int], float] = {}
    warm0: dict[tuple[str, int], float] = {}
    scratch0: dict[tuple[str, int], float] = {}

    for r in rows:
        exp = r["experiment"]
        fmt = str(exp["format"])
        if args.format and fmt != args.format:
            continue
        seed = int(exp["seed"])
        cond = str(exp["condition"])
        wr = float(r["win_rate"])
        key = (fmt, seed)
        if cond == "bc_only":
            bc[key] = wr
        elif cond == "rl_warm" and exp.get("rl_update") == 0:
            warm0[key] = wr
        elif cond == "rl_scratch" and exp.get("rl_update") == 0:
            scratch0[key] = wr

    keys = sorted(set(bc) | set(warm0) | set(scratch0))
    if not keys:
        raise SystemExit("No (format, seed) keys after filter.")

    print(f"sweep_id={args.sweep_id}  rows_used={len(rows)}")
    print(f"{'format':<22} {'seed':>4}  {'bc_only':>8}  {'warm@0':>8}  {'Δ(w-b)':>8}  {'scratch@0':>10}  {'note'}")
    out_rows: list[dict[str, Any]] = []
    for fmt, seed in keys:
        b = bc.get((fmt, seed))
        w = warm0.get((fmt, seed))
        s = scratch0.get((fmt, seed))
        if b is None or w is None:
            note = "missing bc or warm@0"
            delta = ""
        else:
            d = w - b
            delta = f"{d:+.4f}"
            note = ""
            if abs(w - b) > args.warn_delta:
                note = f"|Δ|>{args.warn_delta} — check ckpt / duplicate rows"
        print(
            f"{fmt:<22} {seed:4d}  "
            f"{(f'{b:.4f}' if b is not None else '  —   '):>8}  "
            f"{(f'{w:.4f}' if w is not None else '  —   '):>8}  "
            f"{str(delta):>8}  "
            f"{(f'{s:.4f}' if s is not None else '    —     '):>10}  "
            f"{note}"
        )
        out_rows.append(
            {
                "format": fmt,
                "seed": seed,
                "bc_only": b,
                "rl_warm_update0": w,
                "delta_warm_minus_bc": (w - b) if b is not None and w is not None else None,
                "rl_scratch_update0": s,
            }
        )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
