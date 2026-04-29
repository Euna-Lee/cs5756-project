"""Action space definition and battle-order helpers.

Action indices:
  0 – 3   : Use move slot 0 – 3 of the active Pokémon
  4 – 8   : Switch to bench Pokémon slot 0 – 4

Total: ACTION_DIM = 9
"""

from __future__ import annotations

from typing import Any, List, Optional

ACTION_DIM = 9
N_MOVE_ACTIONS = 4
N_SWITCH_ACTIONS = 5


def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Get a field from either a plain dict (key lookup) or an object (getattr)."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def action_to_move(action: int, battle: Any) -> Any:
    """Convert an integer action into a poke-env order.

    Parameters
    ----------
    action:
        Integer in [0, ACTION_DIM).
    battle:
        A poke-env ``Battle`` object that owns a ``create_order`` method and
        exposes ``available_moves`` / ``available_switches``.

    Returns
    -------
    A poke-env BattleOrder, or None if no valid order can be constructed.
    """
    available_moves: List[Any] = list(_get_attr(battle, "available_moves", []) or [])
    available_switches: List[Any] = list(_get_attr(battle, "available_switches", []) or [])

    if action < N_MOVE_ACTIONS:
        if action < len(available_moves):
            return battle.create_order(available_moves[action])
    else:
        switch_idx = action - N_MOVE_ACTIONS
        if switch_idx < len(available_switches):
            return battle.create_order(available_switches[switch_idx])

    # Fallback: pick the first available legal action
    if available_moves:
        return battle.create_order(available_moves[0])
    if available_switches:
        return battle.create_order(available_switches[0])
    return None


def get_action_mask(battle: Any) -> List[bool]:
    """Return a boolean mask of valid actions for the current battle state.

    Parameters
    ----------
    battle:
        A poke-env ``Battle`` object or a compatible mock dict.

    Returns
    -------
    List[bool] of length ACTION_DIM.  True = action is legal.
    """
    available_moves: List[Any] = list(_get_attr(battle, "available_moves", []) or [])
    available_switches: List[Any] = list(_get_attr(battle, "available_switches", []) or [])
    force_switch: bool = bool(_get_attr(battle, "force_switch", False))
    trapped: bool = bool(_get_attr(battle, "trapped", False))

    mask = [False] * ACTION_DIM

    if not force_switch:
        for i in range(min(len(available_moves), N_MOVE_ACTIONS)):
            mask[i] = True

    if not trapped:
        for i in range(min(len(available_switches), N_SWITCH_ACTIONS)):
            mask[N_MOVE_ACTIONS + i] = True

    # Always ensure at least one action is legal (should not happen in practice)
    if not any(mask):
        mask[0] = True

    return mask
