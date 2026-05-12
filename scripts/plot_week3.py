#!/usr/bin/env python3
"""Plot Week 3 eval curves from runs/week3_sweep_*/eval.jsonl (needs --extra-json from week3_sweep)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--sweep-id", type=int, default=None, help="Filter experiment.sweep_id (default: all in file)")
    p.add_argument(
        "--eval-opponent",
        type=str,
        default="random",
        choices=("random", "heuristic", "any"),
        help="Filter eval rows by top-level `opponent` (use heuristic after reval_heuristic.py; any = no filter).",
    )
    p.add_argument(
        "--x-axis",
        type=str,
        default="rl_train_battles",
        choices=("rl_update", "rl_train_battles", "rl_train_env_steps"),
        help="Horizontal axis for RL curves: PPO update index, cumulative RL *training* battles, or env steps "
        "(from checkpoint; requires eval after train_rl writes cumulative fields).",
    )
    p.add_argument(
        "--battles-per-update",
        type=int,
        default=None,
        help="Override for inferring train battles when eval rows lack cumulative_train_battles / rl_train_battles.",
    )
    p.add_argument(
        "--dump-curve-csv",
        type=Path,
        default=None,
        help="Optional path: write one row per (format, condition, x) with mean/std win rate for the report.",
    )
    p.add_argument("--out", type=Path, default=None, help="PNG path (default: next to eval jsonl)")
    return p.parse_args()


def _manifest_bpu(eval_path: Path, sweep_id: int | None) -> int | None:
    man = eval_path.parent / "sweep_manifest.json"
    if not man.is_file():
        return None
    try:
        data = json.loads(man.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if sweep_id is not None and data.get("sweep_id") != sweep_id:
        return None
    v = data.get("battles_per_update")
    return int(v) if v is not None else None


def _infer_bpu(rows: list[dict], manifest_bpu: int | None, override: int | None) -> int | None:
    if override is not None:
        return override
    for r in rows:
        exp = r.get("experiment")
        if isinstance(exp, dict) and exp.get("battles_per_update") is not None:
            return int(exp["battles_per_update"])
    return manifest_bpu


def _x_from_exp(exp: dict, x_axis: str, *, bpu_fallback: int | None) -> float:
    if x_axis == "rl_update":
        return float(exp["rl_update"])
    if x_axis == "rl_train_battles":
        v = exp.get("cumulative_train_battles")
        if v is not None:
            return float(v)
        v2 = exp.get("rl_train_battles")
        if v2 is not None:
            return float(v2)
        u = exp.get("rl_update")
        if u is None or bpu_fallback is None:
            raise KeyError("Cannot resolve RL train battles (need cumulative_train_battles, rl_train_battles, or "
                           "rl_update plus battles_per_update / --battles-per-update / sweep_manifest.json).")
        return float(int(u) * bpu_fallback)
    if x_axis == "rl_train_env_steps":
        v = exp.get("cumulative_train_env_steps")
        if v is None:
            raise KeyError(
                "cumulative_train_env_steps missing in experiment (re-eval with current eval.py after RL "
                "checkpoints include step counts, or use --x-axis rl_train_battles)."
            )
        return float(v)
    raise ValueError(x_axis)


def _x_label(x_axis: str) -> str:
    if x_axis == "rl_update":
        return "RL update (PPO-style outer step)"
    if x_axis == "rl_train_battles":
        return "Cumulative RL training battles (before checkpoint eval)"
    if x_axis == "rl_train_env_steps":
        return "Cumulative RL training env steps (transitions logged at train time)"
    return x_axis


def main() -> None:
    args = _parse_args()
    if not args.eval_jsonl.is_file():
        raise SystemExit(
            f"Missing eval file: {args.eval_jsonl}\n"
            "Use the sweep folder that matches your w3-<id>-* run dirs (same <id>), "
            "and only after `python scripts/week3_sweep.py --execute` (dry-run folders have no eval). "
            "If the file exists but has only one `sweep_meta` line, the sweep is still running or failed before the first eval."
        )
    rows: list[dict] = []
    with args.eval_jsonl.open(encoding="utf-8") as f:
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
            if not isinstance(rec.get("experiment"), dict):
                continue
            exp = rec.get("experiment")
            if not isinstance(exp, dict):
                continue
            if args.sweep_id is not None and exp.get("sweep_id") != args.sweep_id:
                continue
            top_opp = rec.get("opponent")
            if args.eval_opponent == "random":
                if top_opp == "heuristic":
                    continue
            elif args.eval_opponent == "heuristic":
                if top_opp != "heuristic":
                    continue
            rows.append(rec)

    if not rows:
        raise SystemExit("No matching eval rows (need experiment JSON from week3_sweep / eval --extra-json).")

    manifest_bpu = _manifest_bpu(args.eval_jsonl, args.sweep_id)
    bpu = _infer_bpu(rows, manifest_bpu, args.battles_per_update)

    # RL curves: (format, condition, rl_update) -> win rates; rl_meta stores one experiment dict per key
    rl_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    rl_meta: dict[tuple[str, str, int], dict] = {}
    bc_by_fmt: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        exp = r["experiment"]
        fmt = str(exp["format"])
        cond = str(exp["condition"])
        wr = float(r["win_rate"])
        if cond == "bc_only":
            bc_by_fmt[fmt].append(wr)
        else:
            u = exp.get("rl_update")
            if u is None:
                continue
            ui = int(u)
            key = (fmt, cond, ui)
            rl_groups[key].append(wr)
            if key not in rl_meta:
                rl_meta[key] = exp

    formats = sorted({f for (f, _, _) in rl_groups.keys()} | set(bc_by_fmt.keys()))
    nfmt = len(formats)
    fig, axes = plt.subplots(1, nfmt, figsize=(5 * nfmt, 4), squeeze=False)
    csv_rows: list[dict] = []

    for ax, fmt in zip(axes[0], formats):
        updates = sorted({u for (f, _, u) in rl_groups if f == fmt})
        for cond, color, label in (
            ("rl_scratch", "C0", "RL scratch"),
            ("rl_warm", "C1", "RL warm (BC)"),
        ):
            points: list[tuple[float, float, float]] = []  # (x, mean, std)
            for u in updates:
                key = (fmt, cond, u)
                if key not in rl_groups:
                    continue
                exp = rl_meta[key]
                try:
                    xv = _x_from_exp(exp, args.x_axis, bpu_fallback=bpu)
                except KeyError as e:
                    raise SystemExit(f"{e.args[0]} (format={fmt}, condition={cond}, rl_update={u}).") from e
                vals = rl_groups[key]
                mean = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                    std = var**0.5
                else:
                    std = 0.0
                points.append((xv, mean, std))
                csv_rows.append(
                    {
                        "format": fmt,
                        "condition": cond,
                        "rl_update": u,
                        "x_axis": args.x_axis,
                        "x_value": xv,
                        "mean_win_rate": mean,
                        "std_win_rate": std,
                        "n_seeds": len(vals),
                    }
                )
            points.sort(key=lambda t: t[0])
            if points:
                xs = [p[0] for p in points]
                means = [p[1] for p in points]
                stds = [p[2] for p in points]
                ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=label, color=color)
        if fmt in bc_by_fmt:
            vals = bc_by_fmt[fmt]
            m = sum(vals) / len(vals)
            bc_label = f"BC only (mean={m:.2f}, offline)"
            if args.x_axis == "rl_update":
                bc_label = f"BC only (mean={m:.2f})"
            ax.axhline(m, color="C2", linestyle="--", label=bc_label)
        ax.set_xlabel(_x_label(args.x_axis))
        ax.set_ylabel("Win rate (eval)")
        title = fmt
        if args.eval_opponent == "heuristic":
            title = f"{fmt} (vs heuristic)"
        elif args.eval_opponent == "random":
            title = f"{fmt} (vs random)"
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = args.out or args.eval_jsonl.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    if args.dump_curve_csv:
        args.dump_curve_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_curve_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["format", "condition", "rl_update", "x_axis", "x_value", "mean_win_rate", "std_win_rate", "n_seeds"],
            )
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Wrote {args.dump_curve_csv}")


if __name__ == "__main__":
    main()
