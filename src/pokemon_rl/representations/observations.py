from __future__ import annotations

from typing import Any, Mapping, Sequence


def battle_snapshot(battle: Any) -> dict[str, Any]:
    """
    Best-effort JSON-serializable snapshot of a poke-env Battle.

    This is meant for Week 1 data collection: store something stable, easy to inspect,
    and sufficient to build encoders later. Keep it *pure* (no side effects).
    """
    snap: dict[str, Any] = {
        "battle_tag": getattr(battle, "battle_tag", None),
        "turn": getattr(battle, "turn", None),
        "format": getattr(battle, "format", None),
        "finished": getattr(battle, "finished", None),
        "won": getattr(battle, "won", None),
        "force_switch": getattr(battle, "force_switch", None),
        "can_tera": getattr(battle, "can_tera", None),
        "maybe_trapped": getattr(battle, "maybe_trapped", None),
        "trapped": getattr(battle, "trapped", None),
        "weather": _stringify(getattr(battle, "weather", None)),
        "fields": _mapping_to_json(getattr(battle, "fields", None)),
        "side_conditions": _mapping_to_json(getattr(battle, "side_conditions", None)),
    }

    # Active Pokémon (self + opponent)
    snap["active"] = _pokemon_to_json(getattr(battle, "active_pokemon", None))
    snap["opponent_active"] = _pokemon_to_json(getattr(battle, "opponent_active_pokemon", None))

    # Team info (partial observability: opponent team is usually incomplete)
    snap["team"] = _pokemon_map_to_json(getattr(battle, "team", None))
    snap["opponent_team"] = _pokemon_map_to_json(getattr(battle, "opponent_team", None))

    # Immediate action affordances (useful for BC + debugging)
    snap["available_moves"] = _moves_to_json(getattr(battle, "available_moves", None))
    snap["available_switches"] = _switches_to_json(getattr(battle, "available_switches", None))

    # Game state context that tends to matter for action mapping
    snap["opponent_can_switch"] = getattr(battle, "opponent_can_switch", None)
    snap["rating"] = getattr(battle, "rating", None)

    return snap


def _pokemon_map_to_json(team: Any) -> list[dict[str, Any]]:
    # poke-env uses dict[str, Pokemon] for team/opponent_team.
    if team is None:
        return []
    if isinstance(team, Mapping):
        pokes = list(team.values())
    elif isinstance(team, Sequence):
        pokes = list(team)
    else:
        return []
    return [_pokemon_to_json(p) for p in pokes]


def _pokemon_to_json(p: Any) -> dict[str, Any] | None:
    if p is None:
        return None
    return {
        "species": getattr(p, "species", None),
        "name": getattr(p, "name", None),
        "level": getattr(p, "level", None),
        "current_hp_fraction": getattr(p, "current_hp_fraction", None),
        "fainted": getattr(p, "fainted", None),
        "status": _stringify(getattr(p, "status", None)),
        "type_1": _stringify(getattr(p, "type_1", None)),
        "type_2": _stringify(getattr(p, "type_2", None)),
        "ability": getattr(p, "ability", None),
        "item": getattr(p, "item", None),
        "boosts": _mapping_to_json(getattr(p, "boosts", None)),
        "active": getattr(p, "active", None),
    }


def _moves_to_json(moves: Any) -> list[dict[str, Any]]:
    if not moves:
        return []
    out: list[dict[str, Any]] = []
    for m in list(moves)[:4]:
        # poke-env Move fields are backed by Showdown move entries.
        # Some entries may miss optional keys (e.g. "priority"),
        # and the corresponding @property can raise KeyError.
        # We catch all errors to keep logging robust.
        out.append(
            {
                "id": _safe_attr(m, "id"),
                "name": _safe_attr(m, "name"),
                "type": _stringify(_safe_attr(m, "type")),
                "base_power": _safe_attr(m, "base_power"),
                "accuracy": _safe_attr(m, "accuracy"),
                "priority": _safe_attr(m, "priority"),
                "current_pp": _safe_attr(m, "current_pp"),
            }
        )
    return out


def _switches_to_json(switches: Any) -> list[dict[str, Any]]:
    if not switches:
        return []
    out: list[dict[str, Any]] = []
    for p in list(switches)[:6]:
        out.append(
            {
                "species": getattr(p, "species", None),
                "current_hp_fraction": getattr(p, "current_hp_fraction", None),
                "fainted": getattr(p, "fainted", None),
                "status": _stringify(getattr(p, "status", None)),
            }
        )
    return out


def _mapping_to_json(m: Any) -> dict[str, Any]:
    if not m:
        return {}
    if not isinstance(m, Mapping):
        return {}
    return {str(k): _stringify(v) for k, v in m.items()}


def _stringify(x: Any) -> Any:
    if x is None:
        return None
    # poke-env uses enums for many fields; str() tends to be stable and JSON-safe.
    if isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _safe_attr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr, None)
    except Exception:
        return None

