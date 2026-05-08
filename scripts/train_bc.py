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

    run_id = f"bc-{args.format}-{int(time.time())}"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    model = MaskedPolicyNet(obs_dim=obs_dim, hidden=args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
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
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", acc, epoch)
        print(f"epoch {epoch+1}/{args.epochs} val_loss={val_loss:.4f} val_acc={acc:.3f}")

        ckpt_path = run_dir / f"policy_epoch{epoch+1}.pt"
        torch.save(
            {
                "format": args.format,
                "obs_dim": obs_dim,
                "action_space_size": ACTION_SPACE_SIZE,
                "model_state_dict": model.state_dict(),
                "hidden": args.hidden,
            },
            ckpt_path,
        )

    print(f"Saved checkpoints under: {run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
