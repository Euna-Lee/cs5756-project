"""Observation encoding for Pokémon battles.

Converts a poke-env Battle object into a fixed-size numpy array suitable
for neural network input.  The encoding is deterministic and covers both
the player's side and the opponent's visible information.

Observation layout (total OBS_DIM = 384 floats):
  [0:51]     Active Pokémon (self) – HP, status, types, base stats
  [51:147]   Active Pokémon moves (self) – 4 × 24 features
  [147:192]  Bench Pokémon (self) – 5 × 9 features
  [192:243]  Active Pokémon (opponent) – same layout as self active
  [243:339]  Active Pokémon moves (opponent) – same layout as self moves
  [339:384]  Bench Pokémon (opponent) – same layout as self bench
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POKEMON_TYPES: List[str] = [
    "bug", "dark", "dragon", "electric", "fairy", "fighting",
    "fire", "flying", "ghost", "grass", "ground", "ice",
    "normal", "poison", "psychic", "rock", "steel", "water",
]
N_TYPES = len(POKEMON_TYPES)  # 18

# Status conditions (index 0 = no status)
STATUS_LIST: List[Optional[str]] = [None, "brn", "frz", "par", "psn", "tox", "slp"]
N_STATUS = len(STATUS_LIST)  # 7

MOVE_CATEGORIES: List[str] = ["physical", "special", "status"]
N_MOVE_CATS = len(MOVE_CATEGORIES)  # 3

N_BASE_STATS = 6   # hp, atk, def, spa, spd, spe
MAX_BASE_STAT = 255.0
MAX_BASE_POWER = 250.0

N_MOVES_PER_POKEMON = 4
N_BENCH_SLOTS = 5  # each team has up to 6; 1 active + 5 bench

# Per-entity dimensions
ACTIVE_DIM = 1 + N_STATUS + N_TYPES + (N_TYPES + 1) + N_BASE_STATS  # 51
#             hp  status   type1    type2 (+ "none")  stats
MOVE_DIM = N_TYPES + 1 + 1 + N_MOVE_CATS + 1  # 24
#          type  power acc  category           pp_frac
BENCH_DIM = 1 + N_STATUS + 1  # 9
#           hp  status  alive

# Total observation dimension
OBS_DIM = (ACTIVE_DIM + N_MOVES_PER_POKEMON * MOVE_DIM + N_BENCH_SLOTS * BENCH_DIM) * 2  # 384

# ---------------------------------------------------------------------------
# Look-up tables
# ---------------------------------------------------------------------------

TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(POKEMON_TYPES)}
STATUS_TO_IDX: Dict[Optional[str], int] = {s: i for i, s in enumerate(STATUS_LIST)}
CATEGORY_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(MOVE_CATEGORIES)}


def _type_name(t: Any) -> Optional[str]:
    """Normalise a poke-env PokemonType (or None) to a lowercase string."""
    if t is None:
        return None
    if isinstance(t, str):
        return t.lower()
    # poke-env PokemonType enum – use .name attribute
    return t.name.lower()


def _status_name(s: Any) -> Optional[str]:
    """Normalise a poke-env Status (or None) to a lowercase string."""
    if s is None:
        return None
    if isinstance(s, str):
        return s.lower()
    return s.name.lower()


def _category_name(c: Any) -> str:
    """Normalise a poke-env MoveCategory to a lowercase string."""
    if isinstance(c, str):
        return c.lower()
    return c.name.lower()


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_type_onehot(type_name: Optional[str]) -> np.ndarray:
    """18-dim one-hot for a known type, all-zeros if unknown/None."""
    vec = np.zeros(N_TYPES, dtype=np.float32)
    if type_name is not None:
        idx = TYPE_TO_IDX.get(type_name)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _encode_type2_onehot(type_name: Optional[str]) -> np.ndarray:
    """(18+1)-dim one-hot; last position = 'no second type'."""
    vec = np.zeros(N_TYPES + 1, dtype=np.float32)
    if type_name is None:
        vec[N_TYPES] = 1.0
    else:
        idx = TYPE_TO_IDX.get(type_name)
        if idx is not None:
            vec[idx] = 1.0
        else:
            vec[N_TYPES] = 1.0
    return vec


def _encode_status(status_name: Optional[str]) -> np.ndarray:
    """7-dim one-hot for status condition."""
    vec = np.zeros(N_STATUS, dtype=np.float32)
    idx = STATUS_TO_IDX.get(status_name, 0)
    vec[idx] = 1.0
    return vec


def _encode_base_stats(base_stats: Dict[str, int]) -> np.ndarray:
    """6-dim vector of normalised base stats."""
    keys = ["hp", "atk", "def", "spa", "spd", "spe"]
    return np.array(
        [base_stats.get(k, 0) / MAX_BASE_STAT for k in keys],
        dtype=np.float32,
    )


def encode_active_pokemon(pokemon: Any) -> np.ndarray:
    """Encode a poke-env Pokemon object into a vector of size ACTIVE_DIM (51).

    Works with real poke-env Pokemon objects or mock dicts with the same fields.
    """
    if pokemon is None:
        return np.zeros(ACTIVE_DIM, dtype=np.float32)

    # Support both poke-env objects and plain dicts (for tests)
    if isinstance(pokemon, dict):
        hp_frac = float(pokemon.get("current_hp_fraction", 1.0))
        status = _status_name(pokemon.get("status", None))
        type1 = _type_name(pokemon.get("type_1", None))
        type2 = _type_name(pokemon.get("type_2", None))
        base_stats: Dict[str, int] = pokemon.get("base_stats", {})
    else:
        hp_frac = float(getattr(pokemon, "current_hp_fraction", 1.0))
        status = _status_name(getattr(pokemon, "status", None))
        type1 = _type_name(getattr(pokemon, "type_1", None))
        type2 = _type_name(getattr(pokemon, "type_2", None))
        base_stats = getattr(pokemon, "base_stats", {})

    parts = [
        np.array([hp_frac], dtype=np.float32),
        _encode_status(status),
        _encode_type_onehot(type1),
        _encode_type2_onehot(type2),
        _encode_base_stats(base_stats),
    ]
    return np.concatenate(parts)


def encode_move(move: Any) -> np.ndarray:
    """Encode a poke-env Move object into a vector of size MOVE_DIM (24).

    A zero vector represents a missing / unknown move.
    """
    if move is None:
        return np.zeros(MOVE_DIM, dtype=np.float32)

    if isinstance(move, dict):
        move_type = _type_name(move.get("type", None))
        base_power = float(move.get("base_power", 0))
        accuracy = move.get("accuracy", 1.0)
        category = _category_name(move.get("category", "status"))
        current_pp = float(move.get("current_pp", 1))
        max_pp = float(move.get("max_pp", 1))
    else:
        move_type = _type_name(getattr(move, "type", None))
        base_power = float(getattr(move, "base_power", 0))
        accuracy = getattr(move, "accuracy", 1.0)
        category = _category_name(getattr(move, "category", "status"))
        current_pp = float(getattr(move, "current_pp", 1))
        max_pp = float(getattr(move, "max_pp", 1))

    # Accuracy can be True (always hits) or a float
    if accuracy is True or accuracy is None:
        acc_val = 1.0
    else:
        acc_val = float(accuracy)

    pp_frac = current_pp / max_pp if max_pp > 0 else 0.0

    cat_vec = np.zeros(N_MOVE_CATS, dtype=np.float32)
    cat_idx = CATEGORY_TO_IDX.get(category, 2)  # default to status
    cat_vec[cat_idx] = 1.0

    parts = [
        _encode_type_onehot(move_type),
        np.array([base_power / MAX_BASE_POWER, acc_val], dtype=np.float32),
        cat_vec,
        np.array([pp_frac], dtype=np.float32),
    ]
    return np.concatenate(parts)


def encode_bench_pokemon(pokemon: Any) -> np.ndarray:
    """Encode a benched Pokémon into a vector of size BENCH_DIM (9)."""
    if pokemon is None:
        return np.zeros(BENCH_DIM, dtype=np.float32)

    if isinstance(pokemon, dict):
        hp_frac = float(pokemon.get("current_hp_fraction", 0.0))
        status = _status_name(pokemon.get("status", None))
        fainted = bool(pokemon.get("fainted", True))
    else:
        hp_frac = float(getattr(pokemon, "current_hp_fraction", 0.0))
        status = _status_name(getattr(pokemon, "status", None))
        fainted = bool(getattr(pokemon, "fainted", True))

    alive = 0.0 if fainted else 1.0

    return np.concatenate([
        np.array([hp_frac], dtype=np.float32),
        _encode_status(status),
        np.array([alive], dtype=np.float32),
    ])


def encode_side(
    active: Any,
    moves: List[Any],
    bench: List[Any],
) -> np.ndarray:
    """Encode one side of the battle (active + moves + bench).

    Returns a vector of size ACTIVE_DIM + N_MOVES*MOVE_DIM + N_BENCH*BENCH_DIM = 192.
    """
    active_vec = encode_active_pokemon(active)

    move_vecs = []
    for i in range(N_MOVES_PER_POKEMON):
        m = moves[i] if i < len(moves) else None
        move_vecs.append(encode_move(m))

    bench_vecs = []
    for i in range(N_BENCH_SLOTS):
        p = bench[i] if i < len(bench) else None
        bench_vecs.append(encode_bench_pokemon(p))

    return np.concatenate([active_vec] + move_vecs + bench_vecs)


def encode_battle(battle: Any) -> np.ndarray:
    """Encode a poke-env Battle object into a flat numpy array of size OBS_DIM (384).

    Parameters
    ----------
    battle:
        A poke-env ``Battle`` object, or a mock dict with the same structure
        (used in tests without a live server).

    Returns
    -------
    np.ndarray  shape (OBS_DIM,), dtype float32
    """
    if isinstance(battle, dict):
        active_pokemon = battle.get("active_pokemon")
        available_moves = battle.get("available_moves", [])
        team = battle.get("team", {})
        opp_active = battle.get("opponent_active_pokemon")
        opp_moves = battle.get("opponent_moves", [])
        opp_team = battle.get("opponent_team", {})
    else:
        active_pokemon = getattr(battle, "active_pokemon", None)
        available_moves = list(getattr(battle, "available_moves", []))
        team = getattr(battle, "team", {})
        opp_active = getattr(battle, "opponent_active_pokemon", None)
        # Opponent's revealed moves come from the opponent's active Pokémon object
        opp_active_obj = opp_active
        opp_moves = (
            list(opp_active_obj.moves.values())
            if opp_active_obj is not None and hasattr(opp_active_obj, "moves")
            else []
        )
        opp_team = getattr(battle, "opponent_team", {})

    # Build bench lists (exclude active Pokémon)
    def _get_bench(team_dict: Dict, active: Any) -> List[Any]:
        active_name = None
        if active is not None:
            active_name = (
                active.get("species", None)
                if isinstance(active, dict)
                else getattr(active, "species", None)
            )
        bench = []
        for name, poke in team_dict.items():
            if active_name is not None and name == active_name:
                continue
            bench.append(poke)
        return bench

    self_bench = _get_bench(
        team if isinstance(team, dict) else {str(i): p for i, p in enumerate(team)},
        active_pokemon,
    )
    opp_bench = _get_bench(
        opp_team if isinstance(opp_team, dict) else {str(i): p for i, p in enumerate(opp_team)},
        opp_active,
    )

    self_side = encode_side(active_pokemon, available_moves, self_bench)
    opp_side = encode_side(opp_active, opp_moves, opp_bench)

    obs = np.concatenate([self_side, opp_side])
    assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {obs.shape}"
    return obs
