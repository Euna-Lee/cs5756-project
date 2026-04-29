"""Tests for the behaviour cloning trainer."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from src.environment.obs_encoder import OBS_DIM
from src.environment.action_space import ACTION_DIM
from src.training.bc_trainer import BCConfig, BCTrainer, DemonstrationDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dummy_records(n: int = 100, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        obs = rng.uniform(-1, 1, size=OBS_DIM).tolist()
        action = int(rng.integers(0, ACTION_DIM))
        mask = [True] * ACTION_DIM
        records.append({"observation": obs, "action": action, "action_mask": mask})
    return records


# ---------------------------------------------------------------------------
# DemonstrationDataset tests
# ---------------------------------------------------------------------------

class TestDemonstrationDataset:
    def test_len(self):
        records = make_dummy_records(50)
        ds = DemonstrationDataset(records)
        assert len(ds) == 50

    def test_getitem_types(self):
        records = make_dummy_records(10)
        ds = DemonstrationDataset(records)
        obs, action, mask = ds[0]
        assert isinstance(obs, torch.Tensor)
        assert obs.dtype == torch.float32
        assert obs.shape == (OBS_DIM,)
        assert isinstance(action, torch.Tensor)
        assert action.dtype == torch.long
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.shape == (ACTION_DIM,)

    def test_action_in_range(self):
        records = make_dummy_records(20)
        ds = DemonstrationDataset(records)
        for i in range(len(ds)):
            _, action, _ = ds[i]
            assert 0 <= action.item() < ACTION_DIM

    def test_from_jsonl(self, tmp_path):
        records = make_dummy_records(30)
        path = str(tmp_path / "demo.jsonl")
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        ds = DemonstrationDataset.from_jsonl(path)
        assert len(ds) == 30

    def test_missing_mask_defaults_to_all_true(self):
        records = [{"observation": [0.0] * OBS_DIM, "action": 0}]
        ds = DemonstrationDataset(records)
        _, _, mask = ds[0]
        assert mask.all()


# ---------------------------------------------------------------------------
# BCTrainer tests
# ---------------------------------------------------------------------------

class TestBCTrainer:
    def _make_trainer(self, save_dir: str) -> BCTrainer:
        config = BCConfig(
            hidden_sizes=[64, 64],
            learning_rate=1e-3,
            batch_size=16,
            n_epochs=3,
            val_fraction=0.2,
            patience=10,
            device="cpu",
            save_dir=save_dir,
        )
        return BCTrainer(config=config)

    def test_train_returns_history(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        records = make_dummy_records(100)
        ds = DemonstrationDataset(records)
        history = trainer.train(ds)
        assert len(history) > 0
        assert "train_loss" in history[0]
        assert "val_loss" in history[0]
        assert "train_acc" in history[0]

    def test_loss_decreases_over_epochs(self, tmp_path):
        """Loss should generally trend down for a learnable task."""
        trainer = self._make_trainer(str(tmp_path))
        # Create a simple learnable dataset: always action 0
        records = []
        rng = np.random.default_rng(1)
        for _ in range(200):
            obs = rng.uniform(-1, 1, size=OBS_DIM).tolist()
            records.append({"observation": obs, "action": 0})
        ds = DemonstrationDataset(records)
        config = BCConfig(
            hidden_sizes=[64, 64],
            learning_rate=1e-2,
            batch_size=32,
            n_epochs=10,
            val_fraction=0.2,
            patience=20,
            device="cpu",
            save_dir=str(tmp_path),
        )
        trainer = BCTrainer(config=config)
        history = trainer.train(ds)
        first_loss = history[0]["train_loss"]
        last_loss = history[-1]["train_loss"]
        assert last_loss < first_loss

    def test_save_load_roundtrip(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        records = make_dummy_records(60)
        ds = DemonstrationDataset(records)
        trainer.train(ds)

        save_path = str(tmp_path / "bc_test.pt")
        trainer.save(save_path)
        assert os.path.exists(save_path)

        # Create new trainer and load
        trainer2 = self._make_trainer(str(tmp_path))
        trainer2.load(save_path)

        obs = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            l1, _ = trainer.policy(obs)
            l2, _ = trainer2.policy(obs)
        torch.testing.assert_close(l1, l2)

    def test_val_dataset_override(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        train_records = make_dummy_records(80)
        val_records = make_dummy_records(20, seed=99)
        train_ds = DemonstrationDataset(train_records)
        val_ds = DemonstrationDataset(val_records)
        history = trainer.train(train_ds, val_dataset=val_ds)
        assert len(history) > 0

    def test_best_model_saved(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        records = make_dummy_records(80)
        ds = DemonstrationDataset(records)
        trainer.train(ds)
        best_path = os.path.join(str(tmp_path), "bc_best.pt")
        assert os.path.exists(best_path)
