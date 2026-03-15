import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from encoding import ACTION_TABLE

HIDDEN     = 256
OBS_DIM    = 30
ACTION_DIM = len(ACTION_TABLE)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(HIDDEN, ACTION_DIM)
        self.value_head  = nn.Linear(HIDDEN, 1)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        features      = self.trunk(obs)
        logits        = self.policy_head(features)
        masked_logits = logits.masked_fill(~mask, -1e8)
        dist          = Categorical(logits=masked_logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), self.value_head(features)