from __future__ import annotations

import random
from typing import Any

from poke_env.player import Player

from pokemon_rl.representations import legal_action_mask
from pokemon_rl.representations.actions import action_id_to_order


class RandomLegalPlayer(Player):
    """Samples uniformly among legal actions in the fixed 10-action space (move slots + switch slots)."""

    def choose_move(self, battle: Any):  # type: ignore[override]
        mask = legal_action_mask(battle)
        legal = [i for i, v in enumerate(mask) if v]
        if not legal:
            return self.choose_random_move(battle)
        action_id = random.choice(legal)
        return action_id_to_order(self, battle, action_id)
