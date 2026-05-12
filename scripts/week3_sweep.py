#!/usr/bin/env python3
"""
Week 3 experiment driver: BC + RL scratch + RL warm-start, multi-seed / multi-format.

**Default is dry-run** (prints commands only). Pass --execute to run training + eval.

Major choices (no hidden magic):
- One sweep_id per invocation (default: current Unix time) prefixes all run names.
- Training opponent for RL: --train-opponent (random | heuristic).
- Eval opponent: --eval-opponent (often random for curves; use heuristic for "hard" eval).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="Run commands (default: print only)")
    p.add_argument("--sweep-id", type=int, default=None, help="Stable id for this sweep (default: unix time)")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds (e.g. 0,1,2)")
    p.add_argument("--formats", type=str, default="gen9bssfactory,gen9randombattle", help="Comma formats")
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    p.add_argument("--demos-root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "demos")
    p.add_argument("--bc-epochs", type=int, default=5)
    p.add_argument("--rl-updates", type=int, default=50)
    p.add_argument("--battles-per-update", type=int, default=20)
    p.add_argument(
        "--train-opponent",
        type=str,
        default="random",
        choices=("random", "heuristic"),
        help="Opponent during RL training",
    )
    p.add_argument(
        "--eval-opponent",
        type=str,
        default="random",
        choices=("random", "heuristic"),
        help="Opponent during eval (for curves)",
    )
    p.add_argument("--eval-battles", type=int, default=200)
    p.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate RL when update %% eval_every == 0, plus final update",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--rl-lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--force", action="store_true", help="Pass --force to train scripts if re-running same names")
    return p.parse_args()


def _run(cmd: list[str], *, execute: bool, plan_lines: list[str], cwd: Path) -> None:
    line = shlex.join(cmd)
    plan_lines.append(line)
    print(line)
    if execute:
        subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> None:
    args = _parse_args()
    sweep_id = args.sweep_id if args.sweep_id is not None else int(time.time())
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    sweep_dir = args.runs_dir / f"week3_sweep_{sweep_id}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    eval_out = sweep_dir / "eval.jsonl"
    plan_path = sweep_dir / "PLAN.txt"

    # Create eval.jsonl as soon as --execute starts (each eval.py appends; first eval can take many minutes).
    if args.execute and not eval_out.exists():
        eval_out.write_text(
            json.dumps(
                {
                    "kind": "sweep_meta",
                    "event": "started",
                    "sweep_id": sweep_id,
                    "ts": time.time(),
                    "note": "BC/RL eval rows append below as each eval.py completes.",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    eval_updates: list[int] = []
    for u in range(0, args.rl_updates + 1):
        if u % args.eval_every == 0:
            eval_updates.append(u)
    if args.rl_updates not in eval_updates:
        eval_updates.append(args.rl_updates)
    eval_updates = sorted(set(eval_updates))

    plan_lines: list[str] = []

    for fmt in formats:
        for seed in seeds:
            opp_t = "rand" if args.train_opponent == "random" else "heur"
            bc_name = f"w3-{sweep_id}-bc-{fmt}-s{seed}"
            rl_scratch = f"w3-{sweep_id}-rl-scratch-{fmt}-s{seed}-train{opp_t}"
            rl_warm = f"w3-{sweep_id}-rl-warm-{fmt}-s{seed}-train{opp_t}"
            bc_dir = args.runs_dir / bc_name
            bc_final = bc_dir / f"policy_epoch{args.bc_epochs}.pt"
            rl_scratch_dir = args.runs_dir / rl_scratch
            rl_warm_dir = args.runs_dir / rl_warm

            cmd_bc = [
                py,
                str(repo / "scripts" / "train_bc.py"),
                "--format",
                fmt,
                "--seed",
                str(seed),
                "--epochs",
                str(args.bc_epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.bc_lr),
                "--hidden",
                str(args.hidden),
                "--device",
                args.device,
                "--demos-root",
                str(args.demos_root),
                "--runs-dir",
                str(args.runs_dir),
                "--run-name",
                bc_name,
            ]
            if args.force:
                cmd_bc.append("--force")
            _run(cmd_bc, execute=args.execute, plan_lines=plan_lines, cwd=repo)

            exp_bc = {
                "sweep_id": sweep_id,
                "condition": "bc_only",
                "seed": seed,
                "format": fmt,
                "train_opponent": args.train_opponent,
                "eval_opponent": args.eval_opponent,
                "rl_update": None,
                "battles_per_update": args.battles_per_update,
                "rl_train_battles": 0,
            }
            cmd_eval_bc = [
                py,
                str(repo / "scripts" / "eval.py"),
                "--format",
                fmt,
                "--checkpoint",
                str(bc_final),
                "--n-battles",
                str(args.eval_battles),
                "--seed",
                str(seed),
                "--opponent",
                args.eval_opponent,
                "--out",
                str(eval_out),
                "--device",
                args.device,
                "--extra-json",
                json.dumps(exp_bc),
            ]
            _run(cmd_eval_bc, execute=args.execute, plan_lines=plan_lines, cwd=repo)

            cmd_rl_s = [
                py,
                str(repo / "scripts" / "train_rl.py"),
                "--format",
                fmt,
                "--seed",
                str(seed),
                "--updates",
                str(args.rl_updates),
                "--battles-per-update",
                str(args.battles_per_update),
                "--lr",
                str(args.rl_lr),
                "--hidden",
                str(args.hidden),
                "--device",
                args.device,
                "--runs-dir",
                str(args.runs_dir),
                "--opponent",
                args.train_opponent,
                "--run-name",
                rl_scratch,
            ]
            if args.force:
                cmd_rl_s.append("--force")
            _run(cmd_rl_s, execute=args.execute, plan_lines=plan_lines, cwd=repo)

            for u in eval_updates:
                ckpt = rl_scratch_dir / f"policy_update{u}.pt"
                exp_rl = {
                    "sweep_id": sweep_id,
                    "condition": "rl_scratch",
                    "seed": seed,
                    "format": fmt,
                    "train_opponent": args.train_opponent,
                    "eval_opponent": args.eval_opponent,
                    "rl_update": u,
                    "battles_per_update": args.battles_per_update,
                    "rl_train_battles": u * args.battles_per_update,
                }
                cmd_ev = [
                    py,
                    str(repo / "scripts" / "eval.py"),
                    "--format",
                    fmt,
                    "--checkpoint",
                    str(ckpt),
                    "--n-battles",
                    str(args.eval_battles),
                    "--seed",
                    str(seed),
                    "--opponent",
                    args.eval_opponent,
                    "--out",
                    str(eval_out),
                    "--device",
                    args.device,
                    "--extra-json",
                    json.dumps(exp_rl),
                ]
                _run(cmd_ev, execute=args.execute, plan_lines=plan_lines, cwd=repo)

            cmd_rl_w = [
                py,
                str(repo / "scripts" / "train_rl.py"),
                "--format",
                fmt,
                "--seed",
                str(seed),
                "--updates",
                str(args.rl_updates),
                "--battles-per-update",
                str(args.battles_per_update),
                "--lr",
                str(args.rl_lr),
                "--hidden",
                str(args.hidden),
                "--device",
                args.device,
                "--runs-dir",
                str(args.runs_dir),
                "--opponent",
                args.train_opponent,
                "--init-checkpoint",
                str(bc_final),
                "--run-name",
                rl_warm,
            ]
            if args.force:
                cmd_rl_w.append("--force")
            _run(cmd_rl_w, execute=args.execute, plan_lines=plan_lines, cwd=repo)

            for u in eval_updates:
                ckpt = rl_warm_dir / f"policy_update{u}.pt"
                exp_rl = {
                    "sweep_id": sweep_id,
                    "condition": "rl_warm",
                    "seed": seed,
                    "format": fmt,
                    "train_opponent": args.train_opponent,
                    "eval_opponent": args.eval_opponent,
                    "rl_update": u,
                    "battles_per_update": args.battles_per_update,
                    "rl_train_battles": u * args.battles_per_update,
                }
                cmd_ev = [
                    py,
                    str(repo / "scripts" / "eval.py"),
                    "--format",
                    fmt,
                    "--checkpoint",
                    str(ckpt),
                    "--n-battles",
                    str(args.eval_battles),
                    "--seed",
                    str(seed),
                    "--opponent",
                    args.eval_opponent,
                    "--out",
                    str(eval_out),
                    "--device",
                    args.device,
                    "--extra-json",
                    json.dumps(exp_rl),
                ]
                _run(cmd_ev, execute=args.execute, plan_lines=plan_lines, cwd=repo)

    manifest = {
        "sweep_id": sweep_id,
        "sweep_dir": str(sweep_dir),
        "seeds": seeds,
        "formats": formats,
        "bc_epochs": args.bc_epochs,
        "rl_updates": args.rl_updates,
        "battles_per_update": args.battles_per_update,
        "train_opponent": args.train_opponent,
        "eval_opponent": args.eval_opponent,
        "eval_battles": args.eval_battles,
        "eval_updates": eval_updates,
        "eval_jsonl": str(eval_out),
    }
    (sweep_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {plan_path} and {sweep_dir / 'sweep_manifest.json'}")
    print(f"Eval aggregate: {eval_out}")
    if not args.execute:
        print("\nDry-run only. Re-run with --execute to train and evaluate.")


if __name__ == "__main__":
    main()
