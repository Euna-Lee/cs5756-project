"""Behavior Cloning (BC) training loop.

Trains a ``PolicyNetwork`` via supervised learning to imitate an expert bot.
Demonstrations are loaded from a JSONL file produced by the data-collection
script (see ``scripts/collect_demonstrations.py``).

Each record in the JSONL file must contain:
  - ``"observation"``: list[float] of length OBS_DIM
  - ``"action"``:      int in [0, ACTION_DIM)
  - (optional) ``"action_mask"``: list[bool] of length ACTION_DIM

Training uses cross-entropy loss:
    L = -log π_θ(a_expert | s)

Metrics logged per epoch:
  - Training loss
  - Training accuracy (fraction of expert actions matched)
  - Validation loss & accuracy (if a val split is provided)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from src.agents.policy_network import PolicyNetwork
from src.environment.obs_encoder import OBS_DIM
from src.environment.action_space import ACTION_DIM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DemonstrationDataset(Dataset):
    """Loads BC demonstrations from a JSONL file or a list of records."""

    def __init__(self, records: List[Dict]) -> None:
        self.observations = []
        self.actions = []
        self.masks = []

        for rec in records:
            obs = np.array(rec["observation"], dtype=np.float32)
            act = int(rec["action"])
            mask = (
                np.array(rec["action_mask"], dtype=bool)
                if "action_mask" in rec
                else np.ones(ACTION_DIM, dtype=bool)
            )
            self.observations.append(obs)
            self.actions.append(act)
            self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.observations[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )

    @classmethod
    def from_jsonl(cls, path: str) -> "DemonstrationDataset":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BCConfig:
    """Hyper-parameters for behaviour cloning."""

    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 30
    val_fraction: float = 0.1      # fraction of data held out for validation
    weight_decay: float = 1e-4
    patience: int = 5              # early stopping patience (epochs)
    device: str = "cpu"
    save_dir: str = "checkpoints"  # where to save best model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BCTrainer:
    """Trains a PolicyNetwork via behaviour cloning.

    Parameters
    ----------
    config:
        Hyper-parameter configuration.
    pretrained_path:
        Optional checkpoint to warm-start the network weights.
    """

    def __init__(
        self,
        config: Optional[BCConfig] = None,
        pretrained_path: Optional[str] = None,
    ) -> None:
        self.config = config or BCConfig()
        self.device = torch.device(self.config.device)

        self.policy = PolicyNetwork(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=False)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: DemonstrationDataset,
        val_dataset: Optional[DemonstrationDataset] = None,
    ) -> List[Dict[str, float]]:
        """Train the policy for ``config.n_epochs`` epochs.

        Parameters
        ----------
        dataset:
            Full training dataset (optionally split into train/val).
        val_dataset:
            Optional separate validation set.  If None, a fraction of
            ``dataset`` is held out.

        Returns
        -------
        List of per-epoch metric dicts.
        """
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        if val_dataset is None:
            n_val = max(1, int(len(dataset) * cfg.val_fraction))
            n_train = len(dataset) - n_val
            train_set, val_set = random_split(dataset, [n_train, n_val])
        else:
            train_set = dataset
            val_set = val_dataset

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_path = os.path.join(cfg.save_dir, "bc_best.pt")

        for epoch in range(1, cfg.n_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_s": time.time() - t0,
            }
            self.history.append(metrics)

            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            )

            # Early stopping / best-model save
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.policy.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        # Reload best weights
        if os.path.exists(best_path):
            self.policy.load_state_dict(torch.load(best_path, map_location=self.device))

        return self.history

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
    ) -> Tuple[float, float]:
        self.policy.train(train)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.set_grad_enabled(train):
            for obs, actions, masks in loader:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                masks = masks.to(self.device)

                logits, _ = self.policy(obs, action_mask=masks)
                loss = self.criterion(logits, actions)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bs = obs.size(0)
                total_loss += loss.item() * bs
                preds = logits.argmax(dim=-1)
                total_correct += (preds == actions).sum().item()
                total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc
