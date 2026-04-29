#!/usr/bin/env python3
"""Train a behaviour-cloning (BC) policy from expert demonstrations.

Usage
-----
    # First collect demonstrations (mock mode, no server needed):
    python scripts/collect_demonstrations.py --mock --n-battles 500 \\
        --output-path data/expert_demos.jsonl

    # Then train BC:
    python scripts/train_bc.py \\
        --data data/expert_demos.jsonl \\
        --save-dir checkpoints \\
        --n-epochs 50 \\
        --batch-size 64 \\
        --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from src.training.bc_trainer import BCConfig, BCTrainer, DemonstrationDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behaviour cloning policy.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to expert demonstrations JSONL file.")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-plot", type=str, default=None,
                        help="If given, save a loss/accuracy plot to this path.")
    args = parser.parse_args()

    print(f"Loading demonstrations from {args.data} ...")
    dataset = DemonstrationDataset.from_jsonl(args.data)
    print(f"  {len(dataset)} samples loaded.")

    config = BCConfig(
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        val_fraction=args.val_fraction,
        patience=args.patience,
        device=args.device,
        save_dir=args.save_dir,
    )

    trainer = BCTrainer(config=config)
    history = trainer.train(dataset)

    # Save final model
    final_path = os.path.join(args.save_dir, "bc_final.pt")
    trainer.save(final_path)
    print(f"Saved final BC model → {final_path}")

    # Save history
    history_path = os.path.join(args.save_dir, "bc_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history → {history_path}")

    # Optional plot
    if args.output_plot:
        _plot_history(history, args.output_plot)


def _plot_history(history, path: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("BC Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("BC Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved plot → {path}")


if __name__ == "__main__":
    main()
