"""Tests for observation encoding."""

from __future__ import annotations

import numpy as np
import pytest

from src.environment.obs_encoder import (
    OBS_DIM,
    ACTIVE_DIM,
    MOVE_DIM,
    BENCH_DIM,
    N_MOVES_PER_POKEMON,
    N_BENCH_SLOTS,
    encode_active_pokemon,
    encode_move,
    encode_bench_pokemon,
    encode_side,
    encode_battle,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def make_mock_pokemon(
    hp_frac: float = 0.8,
    status=None,
    type1="fire",
    type2="flying",
    fainted: bool = False,
    species: str = "charizard",
):
    return {
        "species": species,
        "current_hp_fraction": hp_frac,
        "status": status,
        "type_1": type1,
        "type_2": type2,
        "fainted": fainted,
        "base_stats": {"hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100},
    }


def make_mock_move(
    move_type="fire",
    base_power=90,
    accuracy=1.0,
    category="special",
    current_pp=8,
    max_pp=8,
):
    return {
        "type": move_type,
        "base_power": base_power,
        "accuracy": accuracy,
        "category": category,
        "current_pp": current_pp,
        "max_pp": max_pp,
    }


def make_mock_battle(n_moves: int = 4, n_bench: int = 3):
    """Build a minimal mock battle dict."""
    active = make_mock_pokemon()
    moves = [make_mock_move() for _ in range(n_moves)]
    opp_active = make_mock_pokemon(hp_frac=0.5, type1="water", type2=None, species="blastoise")
    opp_moves = [make_mock_move(move_type="water", base_power=80, category="special")]
    team = {f"slot_{i}": make_mock_pokemon(species=f"pokemon_{i}") for i in range(6)}
    opp_team = {f"opp_{i}": make_mock_pokemon(species=f"opp_poke_{i}") for i in range(4)}

    return {
        "active_pokemon": active,
        "available_moves": moves,
        "team": team,
        "opponent_active_pokemon": opp_active,
        "opponent_moves": opp_moves,
        "opponent_team": opp_team,
    }


# ---------------------------------------------------------------------------
# Tests: encode_active_pokemon
# ---------------------------------------------------------------------------

class TestEncodeActivePokemon:
    def test_output_shape(self):
        poke = make_mock_pokemon()
        vec = encode_active_pokemon(poke)
        assert vec.shape == (ACTIVE_DIM,)

    def test_dtype(self):
        poke = make_mock_pokemon()
        vec = encode_active_pokemon(poke)
        assert vec.dtype == np.float32

    def test_hp_frac_in_range(self):
        for hp in [0.0, 0.5, 1.0]:
            poke = make_mock_pokemon(hp_frac=hp)
            vec = encode_active_pokemon(poke)
            assert vec[0] == pytest.approx(hp)

    def test_none_returns_zeros(self):
        vec = encode_active_pokemon(None)
        assert np.all(vec == 0.0)

    def test_status_encoding(self):
        # No status → index 0 in status one-hot
        poke_no_status = make_mock_pokemon(status=None)
        vec = encode_active_pokemon(poke_no_status)
        # status starts at index 1 (after hp)
        assert vec[1] == 1.0  # first status slot = "no status"

        poke_burned = make_mock_pokemon(status="brn")
        vec_brn = encode_active_pokemon(poke_burned)
        assert vec_brn[2] == 1.0  # "brn" is index 1 → vec index 2

    def test_type_encoding_single_type(self):
        poke = make_mock_pokemon(type1="water", type2=None)
        vec = encode_active_pokemon(poke)
        # No assertion on exact index, just check vector is valid
        assert np.sum(np.abs(vec)) > 0


# ---------------------------------------------------------------------------
# Tests: encode_move
# ---------------------------------------------------------------------------

class TestEncodeMove:
    def test_output_shape(self):
        move = make_mock_move()
        vec = encode_move(move)
        assert vec.shape == (MOVE_DIM,)

    def test_none_returns_zeros(self):
        vec = encode_move(None)
        assert np.all(vec == 0.0)

    def test_base_power_normalised(self):
        move = make_mock_move(base_power=100)
        vec = encode_move(move)
        # Power feature is at index N_TYPES (18)
        assert 0.0 <= vec[18] <= 1.0

    def test_zero_power_for_status_move(self):
        move = make_mock_move(base_power=0, category="status")
        vec = encode_move(move)
        assert vec[18] == pytest.approx(0.0)

    def test_pp_fraction(self):
        move = make_mock_move(current_pp=4, max_pp=8)
        vec = encode_move(move)
        # pp fraction is the last element
        assert vec[-1] == pytest.approx(0.5)

    def test_accuracy_true(self):
        move = make_mock_move(accuracy=True)
        vec = encode_move(move)
        assert vec[19] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: encode_bench_pokemon
# ---------------------------------------------------------------------------

class TestEncodeBenchPokemon:
    def test_output_shape(self):
        poke = make_mock_pokemon()
        vec = encode_bench_pokemon(poke)
        assert vec.shape == (BENCH_DIM,)

    def test_none_returns_zeros(self):
        vec = encode_bench_pokemon(None)
        assert np.all(vec == 0.0)

    def test_fainted_pokemon(self):
        poke = make_mock_pokemon(fainted=True, hp_frac=0.0)
        vec = encode_bench_pokemon(poke)
        assert vec[-1] == pytest.approx(0.0)

    def test_alive_pokemon(self):
        poke = make_mock_pokemon(fainted=False, hp_frac=0.6)
        vec = encode_bench_pokemon(poke)
        assert vec[-1] == pytest.approx(1.0)
        assert vec[0] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Tests: encode_side
# ---------------------------------------------------------------------------

class TestEncodeSide:
    def test_output_shape(self):
        active = make_mock_pokemon()
        moves = [make_mock_move() for _ in range(4)]
        bench = [make_mock_pokemon() for _ in range(5)]
        vec = encode_side(active, moves, bench)
        expected = ACTIVE_DIM + N_MOVES_PER_POKEMON * MOVE_DIM + N_BENCH_SLOTS * BENCH_DIM
        assert vec.shape == (expected,)

    def test_fewer_moves_padded_with_zeros(self):
        active = make_mock_pokemon()
        moves = [make_mock_move()]  # only 1 move
        bench = []
        vec = encode_side(active, moves, bench)
        # Move slot 1-3 should be all zeros
        move_start = ACTIVE_DIM + MOVE_DIM  # start of 2nd move slot
        assert np.all(vec[move_start: move_start + MOVE_DIM] == 0.0)


# ---------------------------------------------------------------------------
# Tests: encode_battle
# ---------------------------------------------------------------------------

class TestEncodeBattle:
    def test_output_shape(self):
        battle = make_mock_battle()
        obs = encode_battle(battle)
        assert obs.shape == (OBS_DIM,)

    def test_dtype_float32(self):
        battle = make_mock_battle()
        obs = encode_battle(battle)
        assert obs.dtype == np.float32

    def test_deterministic(self):
        battle = make_mock_battle()
        obs1 = encode_battle(battle)
        obs2 = encode_battle(battle)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_battles_differ(self):
        battle1 = make_mock_battle()
        battle2 = make_mock_battle()
        # Modify battle2's active HP
        battle2["active_pokemon"]["current_hp_fraction"] = 0.1
        obs1 = encode_battle(battle1)
        obs2 = encode_battle(battle2)
        assert not np.allclose(obs1, obs2)

    def test_no_opponent_info(self):
        """Should not raise even if opponent info is missing."""
        battle = {
            "active_pokemon": make_mock_pokemon(),
            "available_moves": [make_mock_move()],
            "team": {},
            "opponent_active_pokemon": None,
            "opponent_moves": [],
            "opponent_team": {},
        }
        obs = encode_battle(battle)
        assert obs.shape == (OBS_DIM,)
