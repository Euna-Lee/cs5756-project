"""Tests for BattleLogger and log utilities."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.environment.obs_encoder import OBS_DIM
from src.utils.battle_logger import (
    BattleLogger,
    compute_battle_stats,
    load_demonstrations,
)


# ---------------------------------------------------------------------------
# BattleLogger tests
# ---------------------------------------------------------------------------

class TestBattleLogger:
    def test_creates_file(self, tmp_path):
        log_path = str(tmp_path)
        logger = BattleLogger(log_dir=log_path, filename="test.jsonl")
        logger.close()
        assert os.path.exists(os.path.join(log_path, "test.jsonl"))

    def test_start_battle_returns_id(self, tmp_path):
        logger = BattleLogger(log_dir=str(tmp_path), filename="test.jsonl")
        bid = logger.start_battle()
        assert isinstance(bid, str)
        assert len(bid) > 0
        logger.close()

    def test_log_step_writes_record(self, tmp_path):
        logger = BattleLogger(log_dir=str(tmp_path), filename="test.jsonl")
        logger.start_battle("battle_001")
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        logger.log_step(obs, action=2, reward=0.0, done=False)
        logger.close()

        path = os.path.join(str(tmp_path), "test.jsonl")
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert len(records) == 1
        assert records[0]["battle_id"] == "battle_001"
        assert records[0]["action"] == 2
        assert records[0]["step"] == 0
        assert len(records[0]["observation"]) == OBS_DIM

    def test_step_counter_increments(self, tmp_path):
        logger = BattleLogger(log_dir=str(tmp_path), filename="test.jsonl")
        logger.start_battle("b")
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for _ in range(5):
            logger.log_step(obs, 0, 0.0, False)
        logger.close()

        path = os.path.join(str(tmp_path), "test.jsonl")
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert records[-1]["step"] == 4

    def test_win_counted(self, tmp_path):
        logger = BattleLogger(log_dir=str(tmp_path), filename="test.jsonl")
        logger.start_battle()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        logger.log_step(obs, 0, 30.0, done=True, outcome="win")
        logger.close()
        assert logger.total_wins == 1
        assert logger.total_battles == 1

    def test_loss_counted(self, tmp_path):
        logger = BattleLogger(log_dir=str(tmp_path), filename="test.jsonl")
        logger.start_battle()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        logger.log_step(obs, 0, -30.0, done=True, outcome="loss")
        logger.close()
        assert logger.total_losses == 1

    def test_summary(self, tmp_path):
        logger = BattleLogger(log_dir=str(tmp_path), filename="test.jsonl")
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        for _ in range(3):
            logger.start_battle()
            logger.log_step(obs, 0, 30.0, done=True, outcome="win")

        for _ in range(2):
            logger.start_battle()
            logger.log_step(obs, 0, -30.0, done=True, outcome="loss")

        logger.close()
        summary = logger.summary()
        assert summary["total_battles"] == 5
        assert summary["wins"] == 3
        assert summary["losses"] == 2
        assert summary["win_rate"] == pytest.approx(0.6)

    def test_context_manager(self, tmp_path):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        with BattleLogger(log_dir=str(tmp_path), filename="ctx.jsonl") as logger:
            logger.start_battle("ctx_battle")
            logger.log_step(obs, 1, 0.5, False)
        path = os.path.join(str(tmp_path), "ctx.jsonl")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# load_demonstrations tests
# ---------------------------------------------------------------------------

class TestLoadDemonstrations:
    def _write_jsonl(self, path, records):
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_loads_valid_records(self, tmp_path):
        obs = [0.0] * OBS_DIM
        records = [
            {"observation": obs, "action": i, "battle_id": "b"}
            for i in range(5)
        ]
        path = str(tmp_path / "demo.jsonl")
        self._write_jsonl(path, records)
        loaded = load_demonstrations(path)
        assert len(loaded) == 5

    def test_skips_missing_keys(self, tmp_path):
        records = [
            {"observation": [0.0] * OBS_DIM, "action": 0},
            {"battle_id": "b"},  # missing observation and action → skipped
        ]
        path = str(tmp_path / "partial.jsonl")
        self._write_jsonl(path, records)
        loaded = load_demonstrations(path)
        assert len(loaded) == 1

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        open(path, "w").close()
        loaded = load_demonstrations(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# compute_battle_stats tests
# ---------------------------------------------------------------------------

class TestComputeBattleStats:
    def _write_log(self, tmp_path, battles):
        """battles: list of (battle_id, steps_rewards, outcome)"""
        path = str(tmp_path / "log.jsonl")
        obs = [0.0] * OBS_DIM
        with open(path, "w") as f:
            for bid, rewards, outcome in battles:
                for i, r in enumerate(rewards):
                    done = i == len(rewards) - 1
                    rec = {
                        "battle_id": bid,
                        "step": i,
                        "observation": obs,
                        "action": 0,
                        "reward": r,
                        "done": done,
                        "outcome": outcome if done else None,
                    }
                    f.write(json.dumps(rec) + "\n")
        return path

    def test_win_rate(self, tmp_path):
        battles = [
            ("b1", [0.0, 1.0], "win"),
            ("b2", [0.0, -1.0], "loss"),
            ("b3", [0.0, 1.0], "win"),
        ]
        path = self._write_log(tmp_path, battles)
        stats = compute_battle_stats(path)
        assert stats["n_battles"] == 3
        assert stats["win_rate"] == pytest.approx(2 / 3)

    def test_empty_log(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        open(path, "w").close()
        stats = compute_battle_stats(path)
        assert "error" in stats
