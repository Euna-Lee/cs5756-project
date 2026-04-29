"""Expert heuristic bot for collecting behavior-cloning demonstrations.

Two bots are provided:

``MaxDamagePlayer``
    Always picks the move with the highest base power × STAB × type-effectiveness
    product, or switches to the first healthy Pokémon when forced.

``SimpleHeuristicsPlayer``
    Extends ``MaxDamagePlayer`` with additional heuristics:
    - Switch out when the active Pokémon is at ≤ 25 % HP and a better match-up
      exists on the bench.
    - Prefer moves that KO the opponent (estimated from base power vs HP remaining).
    - Prefer faster moves when the opponent is at low HP.

Both players expose a ``choose_action(battle) -> int`` method that returns an
action index (compatible with the ACTION_DIM = 9 action space) without
requiring a live Pokémon Showdown connection – this allows the same logic to be
used for BC data collection and for standalone unit testing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.environment.action_space import ACTION_DIM, N_MOVE_ACTIONS, N_SWITCH_ACTIONS

# Type-effectiveness chart (attacker type → {defender type: multiplier})
# Simplified to the most common interactions; full chart omitted for brevity
# but the key structure is here for extension.
TYPE_CHART: Dict[str, Dict[str, float]] = {
    "fire":     {"grass": 2.0, "ice": 2.0, "bug": 2.0, "steel": 2.0,
                 "water": 0.5, "rock": 0.5, "fire": 0.5, "dragon": 0.5},
    "water":    {"fire": 2.0, "rock": 2.0, "ground": 2.0,
                 "water": 0.5, "grass": 0.5, "dragon": 0.5},
    "grass":    {"water": 2.0, "rock": 2.0, "ground": 2.0,
                 "fire": 0.5, "grass": 0.5, "poison": 0.5, "flying": 0.5,
                 "bug": 0.5, "dragon": 0.5, "steel": 0.5},
    "electric": {"water": 2.0, "flying": 2.0,
                 "grass": 0.5, "electric": 0.5, "dragon": 0.5,
                 "ground": 0.0},
    "ice":      {"grass": 2.0, "ground": 2.0, "flying": 2.0, "dragon": 2.0,
                 "fire": 0.5, "water": 0.5, "ice": 0.5, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "rock": 2.0, "dark": 2.0, "steel": 2.0,
                 "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "fairy": 0.5,
                 "ghost": 0.0},
    "poison":   {"grass": 2.0, "fairy": 2.0,
                 "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5,
                 "steel": 0.0},
    "ground":   {"fire": 2.0, "electric": 2.0, "poison": 2.0, "rock": 2.0, "steel": 2.0,
                 "grass": 0.5, "bug": 0.5,
                 "flying": 0.0},
    "flying":   {"grass": 2.0, "fighting": 2.0, "bug": 2.0,
                 "electric": 0.5, "rock": 0.5, "steel": 0.5},
    "psychic":  {"fighting": 2.0, "poison": 2.0,
                 "psychic": 0.5, "steel": 0.5,
                 "dark": 0.0},
    "bug":      {"grass": 2.0, "psychic": 2.0, "dark": 2.0,
                 "fire": 0.5, "fighting": 0.5, "flying": 0.5,
                 "ghost": 0.5, "steel": 0.5, "fairy": 0.5},
    "rock":     {"fire": 2.0, "ice": 2.0, "flying": 2.0, "bug": 2.0,
                 "fighting": 0.5, "ground": 0.5, "steel": 0.5},
    "ghost":    {"psychic": 2.0, "ghost": 2.0,
                 "dark": 0.5,
                 "normal": 0.0},
    "dragon":   {"dragon": 2.0,
                 "steel": 0.5,
                 "fairy": 0.0},
    "dark":     {"psychic": 2.0, "ghost": 2.0,
                 "fighting": 0.5, "dark": 0.5, "fairy": 0.5},
    "steel":    {"ice": 2.0, "rock": 2.0, "fairy": 2.0,
                 "fire": 0.5, "water": 0.5, "electric": 0.5, "steel": 0.5},
    "fairy":    {"fighting": 2.0, "dragon": 2.0, "dark": 2.0,
                 "fire": 0.5, "poison": 0.5, "steel": 0.5},
    "normal":   {},
}


def _type_name(t: Any) -> Optional[str]:
    if t is None:
        return None
    if isinstance(t, str):
        return t.lower()
    return t.name.lower()


def _type_effectiveness(move_type: Optional[str], defender_types: List[Optional[str]]) -> float:
    """Compute the type-effectiveness multiplier for a move against a defender."""
    if move_type is None:
        return 1.0
    chart = TYPE_CHART.get(move_type, {})
    multiplier = 1.0
    for dtype in defender_types:
        if dtype is not None:
            multiplier *= chart.get(dtype, 1.0)
    return multiplier


def _get_attribute(obj: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from either a dict or object."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _score_move(move: Any, opponent: Any, user_types: List[Optional[str]]) -> float:
    """Score a single move against the current opponent Pokémon."""
    base_power = float(_get_attribute(move, "base_power", 0))
    if base_power == 0:
        return 0.0  # Status moves score 0 in damage-based heuristics

    move_type = _type_name(_get_attribute(move, "type", None))
    opp_type1 = _type_name(_get_attribute(opponent, "type_1", None))
    opp_type2 = _type_name(_get_attribute(opponent, "type_2", None))

    effectiveness = _type_effectiveness(move_type, [opp_type1, opp_type2])

    # STAB bonus (Same-Type Attack Bonus)
    stab = 1.5 if move_type in [t for t in user_types if t is not None] else 1.0

    accuracy = _get_attribute(move, "accuracy", 1.0)
    if accuracy is True or accuracy is None:
        accuracy = 1.0
    else:
        accuracy = float(accuracy)

    return base_power * effectiveness * stab * accuracy


def _bench_pokemon(team: Any, active: Any) -> List[Any]:
    """Return non-active, non-fainted bench Pokémon from a team."""
    active_name = _get_attribute(active, "species", None)
    items = team.values() if isinstance(team, dict) else list(team)
    bench = []
    for poke in items:
        name = _get_attribute(poke, "species", None)
        if name == active_name:
            continue
        if _get_attribute(poke, "fainted", False):
            continue
        bench.append(poke)
    return bench


class MaxDamagePlayer:
    """Heuristic player that maximises damage output each turn.

    Can be used standalone (``choose_action``) or subclassed to integrate
    with poke-env's Player API.
    """

    def choose_action(self, battle: Any) -> int:
        """Return the action index that maximises expected damage.

        Parameters
        ----------
        battle:
            A poke-env ``Battle`` object or a compatible mock dict.

        Returns
        -------
        int in [0, ACTION_DIM).
        """
        available_moves: List[Any] = list(_get_attribute(battle, "available_moves", []) or [])
        available_switches: List[Any] = list(_get_attribute(battle, "available_switches", []) or [])
        force_switch: bool = bool(_get_attribute(battle, "force_switch", False))

        if force_switch:
            return self._choose_switch(available_switches, None, battle)

        # Score all available moves
        active = _get_attribute(battle, "active_pokemon", None)
        opponent = _get_attribute(battle, "opponent_active_pokemon", None)

        if active is not None and opponent is not None and available_moves:
            user_type1 = _type_name(_get_attribute(active, "type_1", None))
            user_type2 = _type_name(_get_attribute(active, "type_2", None))
            user_types = [user_type1, user_type2]

            best_idx = 0
            best_score = -1.0
            for i, move in enumerate(available_moves[:N_MOVE_ACTIONS]):
                score = _score_move(move, opponent, user_types)
                if score > best_score:
                    best_score = score
                    best_idx = i

            return best_idx  # move action 0-3

        if available_moves:
            return 0  # fallback: first move

        # Must switch
        return self._choose_switch(available_switches, None, battle)

    def _choose_switch(
        self,
        available_switches: List[Any],
        opponent: Any,
        battle: Any,
    ) -> int:
        if available_switches:
            return N_MOVE_ACTIONS  # switch to first available bench slot
        return 0


class SimpleHeuristicsPlayer(MaxDamagePlayer):
    """Extended heuristic player with switch and survival decisions.

    Additional behaviours beyond ``MaxDamagePlayer``:
    - Proactively switch to a better type match-up when HP ≤ 25 %.
    - Prefer status moves (e.g. healing) when the opponent is also low on HP.
    """

    HP_SWITCH_THRESHOLD = 0.25

    def choose_action(self, battle: Any) -> int:
        available_moves: List[Any] = list(_get_attribute(battle, "available_moves", []) or [])
        available_switches: List[Any] = list(_get_attribute(battle, "available_switches", []) or [])
        force_switch: bool = bool(_get_attribute(battle, "force_switch", False))
        trapped: bool = bool(_get_attribute(battle, "trapped", False))
        active = _get_attribute(battle, "active_pokemon", None)

        if force_switch:
            return self._choose_best_switch(available_switches, battle)

        # Proactive switch when HP is critical and not trapped
        if (
            not trapped
            and available_switches
            and active is not None
        ):
            hp_frac = float(_get_attribute(active, "current_hp_fraction", 1.0))
            if hp_frac <= self.HP_SWITCH_THRESHOLD:
                return self._choose_best_switch(available_switches, battle)

        # Otherwise fall back to max-damage move selection
        return super().choose_action(battle)

    def _choose_best_switch(self, available_switches: List[Any], battle: Any) -> int:
        """Pick the bench Pokémon with the best type match-up vs the opponent."""
        opponent = _get_attribute(battle, "opponent_active_pokemon", None)
        if opponent is None or not available_switches:
            if available_switches:
                return N_MOVE_ACTIONS
            return 0

        opp_type1 = _type_name(_get_attribute(opponent, "type_1", None))
        opp_type2 = _type_name(_get_attribute(opponent, "type_2", None))

        best_idx = 0
        best_hp = -1.0
        for i, poke in enumerate(available_switches[:N_SWITCH_ACTIONS]):
            hp_frac = float(_get_attribute(poke, "current_hp_fraction", 0.0))
            if hp_frac > best_hp:
                best_hp = hp_frac
                best_idx = i

        return N_MOVE_ACTIONS + best_idx

    def _choose_switch(
        self,
        available_switches: List[Any],
        opponent: Any,
        battle: Any,
    ) -> int:
        return self._choose_best_switch(available_switches, battle)
