from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from poke_env.player import Player

from pokemon_rl.data.jsonl_steps import encode_obs_tabular
from pokemon_rl.models import MaskedPolicyNet
from pokemon_rl.representations import battle_snapshot, legal_action_mask
from pokemon_rl.representations.actions import action_id_to_order
from pokemon_rl.rewards import terminal_reward


@dataclass
class StepTransition:
    battle_tag: str
    x: torch.Tensor  # (D,)
    mask: torch.Tensor  # (A,) bool
    action: int
    logp: float


@dataclass
class EpisodeTrajectory:
    battle_tag: str
    transitions: List[StepTransition]
    terminal_reward: float
    won: Optional[bool]


class OnPolicyPolicyPlayer(Player):
    """
    A poke-env Player that samples actions from a torch policy network and stores
    transitions for on-policy updates.
    """

    def __init__(
        self,
        *,
        policy: MaskedPolicyNet,
        device: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self.device = device
        self._current: Dict[str, List[StepTransition]] = {}
        self._finished: List[EpisodeTrajectory] = []

    def choose_move(self, battle: Any):  # type: ignore[override]
        battle_tag = getattr(battle, "battle_tag", "")
        obs = battle_snapshot(battle)
        x = encode_obs_tabular(obs).to(self.device)
        mask = torch.tensor(legal_action_mask(battle), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits = self.policy(x.unsqueeze(0)).logits.squeeze(0)
            masked_logits = self.policy.mask_logits(logits.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action, device=self.device)).item())

        self._current.setdefault(battle_tag, []).append(
            StepTransition(
                battle_tag=battle_tag,
                x=x.detach().cpu(),
                mask=mask.detach().cpu(),
                action=action,
                logp=logp,
            )
        )
        return action_id_to_order(self, battle, action)

    def _battle_finished_callback(self, battle):  # type: ignore[override]
        battle_tag = getattr(battle, "battle_tag", "")
        transitions = self._current.pop(battle_tag, [])
        if not transitions:
            return
        self._finished.append(
            EpisodeTrajectory(
                battle_tag=battle_tag,
                transitions=transitions,
                terminal_reward=float(terminal_reward(battle)),
                won=getattr(battle, "won", None),
            )
        )

    def drain_finished(self) -> List[EpisodeTrajectory]:
        out = self._finished
        self._finished = []
        return out

