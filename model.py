import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from encoding import ACTION_TABLE

HIDDEN     = 512
OBS_DIM    = 89   # 12 pawns × 6 + 13 cards + 2 opp hands + deal_round + skip_flag
ACTION_DIM = len(ACTION_TABLE)


def _make_trunk() -> nn.Sequential:
    trunk = nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
    )
    for layer in trunk:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    return trunk


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_trunk  = _make_trunk()
        self.critic_trunk = _make_trunk()
        self.policy_head  = nn.Linear(HIDDEN, ACTION_DIM)
        self.value_head   = nn.Linear(HIDDEN, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_trunk(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        logits        = self.policy_head(self.actor_trunk(obs))
        masked_logits = logits.masked_fill(~mask, -1e8)
        dist          = Categorical(logits=masked_logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), self.value_head(self.critic_trunk(obs))
