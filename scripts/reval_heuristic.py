#!/usr/bin/env python3
"""
Re-run eval with a harder opponent (SimpleHeuristicsPlayer) for checkpoints referenced in a Week 3 eval.jsonl.

Use after a sweep used --eval-opponent random, to get non-saturated curves (esp. gen9randombattle).

Dedupes by (checkpoint path, format, seed). Skips rows without a checkpoint (--random-policy baselines).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-eval-jsonl", type=Path, required=True)
    p.add_argument("--sweep-id", type=int, required=True)
    p.add_argument("--out", type=Path, default=None, help="Default: <same_dir>/eval_heuristic.jsonl")
    p.add_argument("--n-battles", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dry-run", action="store_true", help="Print unique eval jobs only")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable
    out = args.out or args.from_eval_jsonl.parent / "eval_heuristic.jsonl"

    seen: set[tuple[str, str, int]] = set()
    jobs: list[tuple[str, Path, int, dict]] = []

    with args.from_eval_jsonl.open(encoding="utf-8") as f:
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
            if not isinstance(exp, dict) or exp.get("sweep_id") != args.sweep_id:
                continue
            if rec.get("random_policy"):
                continue
            ck = rec.get("checkpoint")
            if not ck:
                continue
            pth = Path(ck)
            if not pth.is_file():
                continue
            fmt = str(exp["format"])
            seed = int(exp["seed"])
            key = (str(pth.resolve()), fmt, seed)
            if key in seen:
                continue
            seen.add(key)
            extra = dict(exp)
            extra["eval_protocol"] = "heuristic"
            extra["reval_from"] = str(args.from_eval_jsonl)
            jobs.append((fmt, pth, seed, extra))

    print(f"Unique heuristic eval jobs: {len(jobs)} -> {out}")
    if args.dry_run:
        for fmt, pth, seed, _ in jobs:
            print(f"  {fmt} seed={seed} {pth}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt, pth, seed, extra in jobs:
        cmd = [
            py,
            str(repo / "scripts" / "eval.py"),
            "--format",
            fmt,
            "--checkpoint",
            str(pth),
            "--n-battles",
            str(args.n_battles),
            "--seed",
            str(seed),
            "--opponent",
            "heuristic",
            "--out",
            str(out),
            "--device",
            args.device,
            "--extra-json",
            json.dumps(extra, default=str),
        ]
        print(subprocess.list2cmdline(cmd))
        subprocess.run(cmd, check=True, cwd=str(repo))

    print(f"\nAppended heuristic evals to {out}")
    print("Plot with plot_week3.py (same --sweep-id); rows include experiment.eval_protocol == heuristic")


if __name__ == "__main__":
    main()
