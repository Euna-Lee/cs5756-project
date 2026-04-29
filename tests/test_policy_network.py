"""Tests for the PolicyNetwork and PPOAgent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.policy_network import PolicyNetwork
from src.agents.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer, RolloutTransition
from src.environment.obs_encoder import OBS_DIM
from src.environment.action_space import ACTION_DIM


# ---------------------------------------------------------------------------
# PolicyNetwork tests
# ---------------------------------------------------------------------------

class TestPolicyNetwork:
    def _make_net(self) -> PolicyNetwork:
        return PolicyNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_sizes=[64, 64])

    def test_forward_output_shapes(self):
        net = self._make_net()
        obs = torch.randn(8, OBS_DIM)
        logits, values = net(obs)
        assert logits.shape == (8, ACTION_DIM)
        assert values.shape == (8,)

    def test_forward_single_obs(self):
        net = self._make_net()
        obs = torch.randn(1, OBS_DIM)
        logits, values = net(obs)
        assert logits.shape == (1, ACTION_DIM)
        assert values.shape == (1,)

    def test_action_mask_applied(self):
        net = self._make_net()
        obs = torch.randn(4, OBS_DIM)
        # Mask: only action 0 is valid
        mask = torch.zeros(4, ACTION_DIM, dtype=torch.bool)
        mask[:, 0] = True
        logits, _ = net(obs, action_mask=mask)
        # All actions except 0 should be very negative
        assert (logits[:, 1:] < -1e8).all()

    def test_get_action_shapes(self):
        net = self._make_net()
        obs = torch.randn(3, OBS_DIM)
        action, log_prob, value = net.get_action(obs)
        assert action.shape == (3,)
        assert log_prob.shape == (3,)
        assert value.shape == (3,)

    def test_get_action_valid_range(self):
        net = self._make_net()
        obs = torch.randn(20, OBS_DIM)
        action, _, _ = net.get_action(obs)
        assert (action >= 0).all() and (action < ACTION_DIM).all()

    def test_deterministic_action_is_argmax(self):
        net = self._make_net()
        obs = torch.randn(5, OBS_DIM)
        logits, _ = net(obs)
        expected = logits.argmax(dim=-1)
        action, _, _ = net.get_action(obs, deterministic=True)
        torch.testing.assert_close(action, expected)

    def test_evaluate_actions_shapes(self):
        net = self._make_net()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randint(0, ACTION_DIM, (10,))
        log_probs, entropy, values = net.evaluate_actions(obs, actions)
        assert log_probs.shape == (10,)
        assert entropy.shape == ()  # scalar
        assert values.shape == (10,)

    def test_log_probs_are_negative(self):
        net = self._make_net()
        obs = torch.randn(5, OBS_DIM)
        actions = torch.randint(0, ACTION_DIM, (5,))
        log_probs, _, _ = net.evaluate_actions(obs, actions)
        assert (log_probs <= 0).all()

    def test_entropy_positive(self):
        net = self._make_net()
        obs = torch.randn(8, OBS_DIM)
        actions = torch.randint(0, ACTION_DIM, (8,))
        _, entropy, _ = net.evaluate_actions(obs, actions)
        assert entropy.item() >= 0.0

    def test_gradient_flows(self):
        net = self._make_net()
        obs = torch.randn(4, OBS_DIM)
        actions = torch.randint(0, ACTION_DIM, (4,))
        log_probs, entropy, values = net.evaluate_actions(obs, actions)
        loss = -log_probs.mean() + 0.5 * values.pow(2).mean() - 0.01 * entropy
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None


# ---------------------------------------------------------------------------
# RolloutBuffer tests
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def _make_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(n_steps=10, obs_dim=OBS_DIM, gamma=0.99, gae_lambda=0.95)

    def _add_transitions(self, buf: RolloutBuffer, n: int = 5) -> None:
        for i in range(n):
            buf.add(RolloutTransition(
                obs=np.zeros(OBS_DIM, dtype=np.float32),
                action=0,
                reward=1.0,
                done=(i == n - 1),
                log_prob=-1.0,
                value=0.5,
            ))

    def test_len(self):
        buf = self._make_buffer()
        assert len(buf) == 0
        self._add_transitions(buf, 5)
        assert len(buf) == 5

    def test_clear(self):
        buf = self._make_buffer()
        self._add_transitions(buf, 5)
        buf.compute_returns_and_advantages(0.0)
        buf.clear()
        assert len(buf) == 0

    def test_advantages_shape(self):
        buf = self._make_buffer()
        self._add_transitions(buf, 8)
        buf.compute_returns_and_advantages(last_value=0.0)
        assert buf.advantages.shape == (8,)
        assert buf.returns.shape == (8,)

    def test_returns_higher_with_positive_rewards(self):
        buf = self._make_buffer()
        self._add_transitions(buf, 5)
        buf.compute_returns_and_advantages(last_value=0.0)
        # With all reward = 1, returns should be positive
        assert buf.returns[0] > 0

    def test_get_tensors(self):
        buf = self._make_buffer()
        self._add_transitions(buf, 6)
        buf.compute_returns_and_advantages(0.0)
        obs_t, acts_t, lp_t, adv_t, ret_t, masks_t = buf.get_tensors("cpu")
        assert obs_t.shape == (6, OBS_DIM)
        assert acts_t.shape == (6,)
        assert adv_t.shape == (6,)
        assert masks_t is None  # no masks provided


# ---------------------------------------------------------------------------
# PPOAgent tests
# ---------------------------------------------------------------------------

class TestPPOAgent:
    def _make_agent(self) -> PPOAgent:
        cfg = PPOConfig(
            hidden_sizes=[64, 64],
            n_steps=16,
            batch_size=8,
            n_epochs=2,
            device="cpu",
        )
        return PPOAgent(cfg)

    def test_act_output_types(self):
        agent = self._make_agent()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action, log_prob, value = agent.act(obs)
        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_act_valid_action_range(self):
        agent = self._make_agent()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _, _ = agent.act(obs)
        assert 0 <= action < ACTION_DIM

    def test_act_with_mask(self):
        agent = self._make_agent()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[2] = True  # only action 2 is valid
        # Deterministic should always pick action 2
        action, _, _ = agent.act(obs, action_mask=mask, deterministic=True)
        assert action == 2

    def test_save_load_roundtrip(self, tmp_path):
        agent = self._make_agent()
        path = str(tmp_path / "test_agent.pt")
        agent.save(path)

        agent2 = self._make_agent()
        agent2.load(path)

        obs = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            l1, v1 = agent.policy(obs)
            l2, v2 = agent2.policy(obs)
        torch.testing.assert_close(l1, l2)

    def test_update_returns_losses(self):
        agent = self._make_agent()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        # Fill buffer with dummy transitions
        for i in range(16):
            agent.buffer.add(RolloutTransition(
                obs=obs,
                action=0,
                reward=float(i % 2),
                done=(i == 15),
                log_prob=-1.0,
                value=0.5,
            ))
        agent.buffer.compute_returns_and_advantages(0.0)
        losses = agent.update()
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses

    def test_pretrained_path_loaded(self, tmp_path):
        """Agent should load weights from a pretrained checkpoint."""
        agent1 = self._make_agent()
        path = str(tmp_path / "pretrained.pt")
        agent1.save(path)

        agent2 = PPOAgent(
            PPOConfig(hidden_sizes=[64, 64], device="cpu"),
            pretrained_path=path,
        )
        obs = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            l1, _ = agent1.policy(obs)
            l2, _ = agent2.policy(obs)
        torch.testing.assert_close(l1, l2)
