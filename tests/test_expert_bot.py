"""Tests for the expert heuristic bots."""

from __future__ import annotations

import pytest

from src.agents.expert_bot import MaxDamagePlayer, SimpleHeuristicsPlayer
from src.environment.action_space import ACTION_DIM, N_MOVE_ACTIONS


# ---------------------------------------------------------------------------
# Mock battle helpers (reuse pattern from test_obs_encoder.py)
# ---------------------------------------------------------------------------

def make_mock_pokemon(
    hp_frac: float = 1.0,
    type1: str = "fire",
    type2=None,
    fainted: bool = False,
    species: str = "test_poke",
):
    return {
        "species": species,
        "current_hp_fraction": hp_frac,
        "fainted": fainted,
        "status": None,
        "type_1": type1,
        "type_2": type2,
        "base_stats": {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
    }


def make_mock_move(move_type: str = "fire", base_power: int = 90, category: str = "special"):
    return {
        "type": move_type,
        "base_power": base_power,
        "accuracy": 1.0,
        "category": category,
        "current_pp": 8,
        "max_pp": 8,
    }


def make_battle(
    moves=None,
    switches=None,
    active_hp: float = 1.0,
    active_type: str = "fire",
    opp_type: str = "water",
    force_switch: bool = False,
    trapped: bool = False,
):
    if moves is None:
        moves = [make_mock_move()]
    if switches is None:
        switches = []

    return {
        "active_pokemon": make_mock_pokemon(hp_frac=active_hp, type1=active_type),
        "available_moves": moves,
        "available_switches": switches,
        "force_switch": force_switch,
        "trapped": trapped,
        "opponent_active_pokemon": make_mock_pokemon(type1=opp_type),
        "team": {},
        "opponent_team": {},
    }


# ---------------------------------------------------------------------------
# MaxDamagePlayer tests
# ---------------------------------------------------------------------------

class TestMaxDamagePlayer:
    def setup_method(self):
        self.player = MaxDamagePlayer()

    def test_returns_valid_action(self):
        battle = make_battle()
        action = self.player.choose_action(battle)
        assert 0 <= action < ACTION_DIM

    def test_chooses_higher_power_move(self):
        moves = [
            make_mock_move(move_type="normal", base_power=40),
            make_mock_move(move_type="normal", base_power=120),
        ]
        battle = make_battle(moves=moves, active_type="normal", opp_type="normal")
        action = self.player.choose_action(battle)
        assert action == 1  # 120 BP move

    def test_force_switch_returns_switch_action(self):
        switches = [make_mock_pokemon(species="poke_0")]
        battle = make_battle(moves=[], switches=switches, force_switch=True)
        action = self.player.choose_action(battle)
        assert action >= N_MOVE_ACTIONS

    def test_no_moves_uses_switch(self):
        switches = [make_mock_pokemon()]
        battle = make_battle(moves=[], switches=switches)
        action = self.player.choose_action(battle)
        assert action >= N_MOVE_ACTIONS

    def test_stab_bonus_preferred(self):
        """A weaker move with STAB should outscore a stronger move without STAB."""
        moves = [
            make_mock_move(move_type="fire", base_power=60),   # STAB (user is fire type)
            make_mock_move(move_type="water", base_power=80),  # no STAB
        ]
        # Fire × fire type = STAB 1.5, so 60*1.5=90 > 80*1.0
        battle = make_battle(moves=moves, active_type="fire", opp_type="normal")
        action = self.player.choose_action(battle)
        assert action == 0  # fire move preferred


# ---------------------------------------------------------------------------
# SimpleHeuristicsPlayer tests
# ---------------------------------------------------------------------------

class TestSimpleHeuristicsPlayer:
    def setup_method(self):
        self.player = SimpleHeuristicsPlayer()

    def test_switches_when_low_hp(self):
        """Should prefer switching when HP ≤ 25% and a switch is available."""
        switches = [make_mock_pokemon(hp_frac=0.9, species="bench_poke")]
        battle = make_battle(
            active_hp=0.1,  # below threshold
            switches=switches,
            trapped=False,
        )
        action = self.player.choose_action(battle)
        assert action >= N_MOVE_ACTIONS

    def test_no_switch_when_trapped(self):
        """When trapped, should not switch even at low HP."""
        switches = [make_mock_pokemon(hp_frac=0.9)]
        battle = make_battle(active_hp=0.1, switches=switches, trapped=True)
        action = self.player.choose_action(battle)
        # Should pick a move action (not a switch)
        assert action < N_MOVE_ACTIONS

    def test_no_switch_at_full_hp(self):
        """Should not switch when HP is high."""
        switches = [make_mock_pokemon()]
        moves = [make_mock_move()]
        battle = make_battle(active_hp=1.0, moves=moves, switches=switches)
        action = self.player.choose_action(battle)
        # With full HP, moves should be preferred
        assert action < N_MOVE_ACTIONS

    def test_force_switch_uses_switch(self):
        switches = [make_mock_pokemon()]
        battle = make_battle(force_switch=True, switches=switches)
        action = self.player.choose_action(battle)
        assert action >= N_MOVE_ACTIONS
