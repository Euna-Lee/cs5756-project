#!/usr/bin/env python3
"""Behavior cloning pretraining from expert JSONL demos."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pokemon_rl.data import JsonlStepDataset, find_step_logs
from pokemon_rl.models import MaskedPolicyNet
from pokemon_rl.representations import ACTION_SPACE_SIZE
from pokemon_rl.utils.run_metadata import common_run_metadata, save_run_manifest


def _collate_step_examples(batch):
    # batch: list[StepExample]
    x = torch.stack([b.x for b in batch], dim=0)
    a = torch.tensor([b.action_id for b in batch], dtype=torch.long)
    m = torch.stack([b.legal_mask for b in batch], dim=0)
    return x, a, m


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", type=str, required=True, help="Showdown format (e.g. gen9bssfactory)")
    p.add_argument("--demos-root", type=Path, default=_REPO_ROOT / "data" / "demos")
    p.add_argument("--runs-dir", type=Path, default=_REPO_ROOT / "runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Directory name under --runs-dir (default: bc-<format>-<timestamp>). Use for reproducible Week 3 sweeps.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing --run-name directory (default: refuse if run_manifest.json exists).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    jsonl_paths = find_step_logs(args.demos_root, args.format)
    if not jsonl_paths:
        raise SystemExit(f"No demos found at {args.demos_root}/{args.format}/*/expert_steps.jsonl")

    ds = JsonlStepDataset(jsonl_paths)
    obs_dim = int(ds[0].x.shape[0])

    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_step_examples
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_step_examples)

    run_id = args.run_name if args.run_name else f"bc-{args.format}-{int(time.time())}"
    run_dir = args.runs_dir / run_id
    if run_dir.exists() and (run_dir / "run_manifest.json").exists() and not args.force:
        raise SystemExit(
            f"Run directory already exists: {run_dir}\n"
            "Use a new --run-name or pass --force to overwrite (not recommended)."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    model = MaskedPolicyNet(obs_dim=obs_dim, hidden=args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    demo_paths: list[str] = []
    for p in jsonl_paths:
        try:
            demo_paths.append(str(p.relative_to(_REPO_ROOT)))
        except ValueError:
            demo_paths.append(str(p))

    def save_ckpt(epoch: int, *, final_val_acc: float | None = None, final_val_loss: float | None = None) -> None:
        meta = common_run_metadata(
            "train_bc.py",
            _REPO_ROOT,
            seed=args.seed,
            extra={
                "run_id": run_id,
                "format": args.format,
                "epochs": args.epochs,
                "epoch_saved": epoch,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden": args.hidden,
                "val_frac": args.val_frac,
                "device": args.device,
                "demos_root": str(args.demos_root),
                "n_demo_files": len(jsonl_paths),
                "demo_files_sample": demo_paths[:30],
                "n_train_examples": n_train,
                "n_val_examples": n_val,
                "final_val_acc": final_val_acc,
                "final_val_loss": final_val_loss,
            },
        )
        ckpt_path = run_dir / f"policy_epoch{epoch}.pt"
        torch.save(
            {
                "format": args.format,
                "obs_dim": obs_dim,
                "action_space_size": ACTION_SPACE_SIZE,
                "model_state_dict": model.state_dict(),
                "hidden": args.hidden,
                "run_metadata": meta,
            },
            ckpt_path,
        )

    # Week 2 exit check: untrained init (same as random policy structure, not uniform-legal baseline)
    save_ckpt(0)

    global_step = 0
    last_val_acc = 0.0
    last_val_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, a, m in train_loader:
            x = x.to(args.device)
            a = a.to(args.device)
            m = m.to(args.device)

            out = model(x).logits
            masked = model.mask_logits(out, m)
            loss = F.cross_entropy(masked, a)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, a, m in val_loader:
                x = x.to(args.device)
                a = a.to(args.device)
                m = m.to(args.device)
                logits = model.mask_logits(model(x).logits, m)
                val_loss += F.cross_entropy(logits, a, reduction="sum").item()
                pred = logits.argmax(dim=-1)
                correct += int((pred == a).sum().item())
                total += int(a.numel())

        val_loss /= max(1, total)
        acc = correct / max(1, total)
        last_val_loss = val_loss
        last_val_acc = acc
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", acc, epoch)
        print(f"epoch {epoch+1}/{args.epochs} val_loss={val_loss:.4f} val_acc={acc:.3f}")

        save_ckpt(epoch + 1, final_val_acc=acc, final_val_loss=val_loss)

    save_run_manifest(
        run_dir / "run_manifest.json",
        common_run_metadata(
            "train_bc.py",
            _REPO_ROOT,
            seed=args.seed,
            extra={
                "run_id": run_id,
                "run_dir": str(run_dir),
                "format": args.format,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden": args.hidden,
                "val_frac": args.val_frac,
                "device": args.device,
                "demos_root": str(args.demos_root),
                "n_demo_files": len(jsonl_paths),
                "demo_files": demo_paths,
                "n_train_examples": n_train,
                "n_val_examples": n_val,
                "final_val_acc": last_val_acc,
                "final_val_loss": last_val_loss,
                "checkpoints": [f"policy_epoch{i}.pt" for i in range(args.epochs + 1)],
            },
        ),
    )
    print(f"Saved checkpoints under: {run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
