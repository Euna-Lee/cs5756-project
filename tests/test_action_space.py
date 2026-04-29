"""Tests for action space mapping and masking."""

from __future__ import annotations

from typing import Any, List

import pytest

from src.environment.action_space import (
    ACTION_DIM,
    N_MOVE_ACTIONS,
    N_SWITCH_ACTIONS,
    get_action_mask,
)


# ---------------------------------------------------------------------------
# Mock battle helper
# ---------------------------------------------------------------------------

class MockMove:
    def __init__(self, name: str):
        self.id = name


class MockPokemon:
    def __init__(self, name: str):
        self.species = name


class MockBattleState:
    """Minimal mock that mimics poke-env Battle attributes."""

    def __init__(
        self,
        n_moves: int = 4,
        n_switches: int = 5,
        force_switch: bool = False,
        trapped: bool = False,
    ) -> None:
        self.available_moves: List[MockMove] = [MockMove(f"move_{i}") for i in range(n_moves)]
        self.available_switches: List[MockPokemon] = [
            MockPokemon(f"poke_{i}") for i in range(n_switches)
        ]
        self.force_switch = force_switch
        self.trapped = trapped

        # create_order records what was ordered
        self._orders: List[Any] = []

    def create_order(self, target: Any) -> Any:
        order = ("order", target)
        self._orders.append(order)
        return order


# ---------------------------------------------------------------------------
# Tests: get_action_mask
# ---------------------------------------------------------------------------

class TestGetActionMask:
    def test_full_mask_all_available(self):
        battle = MockBattleState(n_moves=4, n_switches=5)
        mask = get_action_mask(battle)
        assert len(mask) == ACTION_DIM
        assert all(mask)

    def test_only_moves_available(self):
        battle = MockBattleState(n_moves=3, n_switches=0)
        mask = get_action_mask(battle)
        assert mask[0] is True
        assert mask[1] is True
        assert mask[2] is True
        assert mask[3] is False  # no 4th move
        assert mask[4] is False  # no switch 0

    def test_forced_switch_masks_moves(self):
        battle = MockBattleState(n_moves=4, n_switches=3, force_switch=True)
        mask = get_action_mask(battle)
        # Moves must all be masked
        for i in range(N_MOVE_ACTIONS):
            assert mask[i] is False, f"Move {i} should be masked during force switch"
        # Switches available
        for i in range(3):
            assert mask[N_MOVE_ACTIONS + i] is True
        for i in range(3, N_SWITCH_ACTIONS):
            assert mask[N_MOVE_ACTIONS + i] is False

    def test_trapped_masks_switches(self):
        battle = MockBattleState(n_moves=4, n_switches=5, trapped=True)
        mask = get_action_mask(battle)
        for i in range(N_MOVE_ACTIONS):
            assert mask[i] is True
        for i in range(N_SWITCH_ACTIONS):
            assert mask[N_MOVE_ACTIONS + i] is False

    def test_always_at_least_one_action(self):
        """Even with empty available moves/switches, mask has at least one True."""
        battle = MockBattleState(n_moves=0, n_switches=0)
        mask = get_action_mask(battle)
        assert any(mask)

    def test_mask_length(self):
        battle = MockBattleState()
        mask = get_action_mask(battle)
        assert len(mask) == ACTION_DIM
