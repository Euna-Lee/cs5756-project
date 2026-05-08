from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# Fixed discrete action space:
# - 4 move slots (if available)
# - 6 switch slots (if available)
#
# Total actions = 10. This is intentionally simple and stable across battles.
MAX_MOVES = 4
MAX_SWITCHES = 6
ACTION_SPACE_SIZE = MAX_MOVES + MAX_SWITCHES


@dataclass(frozen=True)
class DecodedAction:
    kind: str  # "move" | "switch"
    index: int  # 0-based within that kind


def decode_action(action_id: int) -> DecodedAction:
    if not (0 <= action_id < ACTION_SPACE_SIZE):
        raise ValueError(f"action_id must be in [0, {ACTION_SPACE_SIZE}), got {action_id}")
    if action_id < MAX_MOVES:
        return DecodedAction(kind="move", index=action_id)
    return DecodedAction(kind="switch", index=action_id - MAX_MOVES)


def action_id_to_label(action_id: int) -> str:
    da = decode_action(action_id)
    if da.kind == "move":
        return f"move:{da.index}"
    return f"switch:{da.index}"


def legal_action_mask(battle: Any) -> list[int]:
    """
    Build a fixed-length {0,1} mask for our discrete action space.

    We intentionally only use `available_moves` and `available_switches` to avoid
    Showdown-protocol quirks and keep this stable for Week 1 logging / BC.
    """
    moves: Sequence[Any] = getattr(battle, "available_moves", ()) or ()
    switches: Sequence[Any] = getattr(battle, "available_switches", ()) or ()

    mask = [0] * ACTION_SPACE_SIZE
    for i in range(min(len(moves), MAX_MOVES)):
        mask[i] = 1
    for j in range(min(len(switches), MAX_SWITCHES)):
        mask[MAX_MOVES + j] = 1
    return mask


def action_id_to_order(player: Any, battle: Any, action_id: int):
    """
    Convert an action_id into a poke-env BattleOrder using `player.create_order`.

    This uses the *current* battle state's `available_moves` / `available_switches`.
    If the action is illegal, falls back to `player.choose_random_move(battle)`.
    """
    da = decode_action(action_id)
    if da.kind == "move":
        moves: Sequence[Any] = getattr(battle, "available_moves", ()) or ()
        if da.index < len(moves):
            return player.create_order(moves[da.index])
        return player.choose_random_move(battle)
    switches: Sequence[Any] = getattr(battle, "available_switches", ()) or ()
    if da.index < len(switches):
        return player.create_order(switches[da.index])
    return player.choose_random_move(battle)


def action_id_to_semantic(
    battle: Any,
    action_id: int,
) -> dict[str, Any]:
    """
    Convert an action_id into a small, JSON-friendly description using the *current*
    battle state's available moves/switches.

    This is primarily for logging. The returned dict is safe to JSON-serialize
    (best-effort: some poke-env objects may stringify).
    """
    da = decode_action(action_id)
    if da.kind == "move":
        moves: Sequence[Any] = getattr(battle, "available_moves", ()) or ()
        if da.index >= len(moves):
            return {"kind": "move", "index": da.index, "available": False}
        m = moves[da.index]
        return {
            "kind": "move",
            "index": da.index,
            "available": True,
            "id": _safe_attr(m, "id"),
            "name": _safe_attr(m, "name"),
            "type": str(_safe_attr(m, "type")),
            "base_power": _safe_attr(m, "base_power"),
            "accuracy": _safe_attr(m, "accuracy"),
            "priority": _safe_attr(m, "priority"),
        }

    switches: Sequence[Any] = getattr(battle, "available_switches", ()) or ()
    if da.index >= len(switches):
        return {"kind": "switch", "index": da.index, "available": False}
    p = switches[da.index]
    return {
        "kind": "switch",
        "index": da.index,
        "available": True,
        "species": _safe_attr(p, "species"),
        "current_hp_fraction": _safe_attr(p, "current_hp_fraction"),
        "status": str(_safe_attr(p, "status")),
    }


def _safe_attr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr, None)
    except Exception:
        return None

