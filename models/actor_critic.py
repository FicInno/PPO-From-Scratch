"""Actor-Critic networks for discrete and bounded continuous actions."""

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layer = nn.Linear(last_dim, hidden_dim)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.Tanh()])
        last_dim = hidden_dim
    output = nn.Linear(last_dim, output_dim)
    nn.init.orthogonal_(output.weight, gain=1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs.

    Continuous policies use a diagonal Gaussian followed by tanh and affine
    scaling to the environment action bounds. The buffer stores the pre-tanh
    action so its log probability can be recomputed without numerical inversion.
    """

    def __init__(
        self,
        observation_dim: int,
        action_space,
        hidden_sizes: Sequence[int] = (64, 64),
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.is_discrete = hasattr(action_space, "n")
        action_dim = int(action_space.n) if self.is_discrete else int(np.prod(action_space.shape))

        self.actor = _mlp(observation_dim, hidden_sizes, action_dim)
        self.critic = _mlp(observation_dim, hidden_sizes, 1)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

        if self.is_discrete:
            self.log_std = None
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
            low = torch.as_tensor(action_space.low, dtype=torch.float32).reshape(-1)
            high = torch.as_tensor(action_space.high, dtype=torch.float32).reshape(-1)
            self.register_buffer("action_scale", (high - low) / 2.0)
            self.register_buffer("action_bias", (high + low) / 2.0)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def policy_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.actor.parameters())
        if self.log_std is not None:
            parameters.append(self.log_std)
        return parameters

    def value_parameters(self) -> list[nn.Parameter]:
        return list(self.critic.parameters())

    def _continuous_distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.actor(observations)
        assert self.log_std is not None
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _tanh_log_det(raw_actions: torch.Tensor) -> torch.Tensor:
        # Stable equivalent of log(1 - tanh(x)^2).
        return (2.0 * (math.log(2.0) - raw_actions - nn.functional.softplus(-2.0 * raw_actions))).sum(-1)

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        values = self.value(observations)
        if self.is_discrete:
            distribution = Categorical(logits=self.actor(observations))
            actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            return actions, actions, distribution.log_prob(actions), values

        distribution = self._continuous_distribution(observations)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        squashed = torch.tanh(raw_actions)
        env_actions = squashed * self.action_scale + self.action_bias
        log_probs = (
            distribution.log_prob(raw_actions).sum(-1)
            - self._tanh_log_det(raw_actions)
            - torch.log(self.action_scale).sum()
        )
        return env_actions, raw_actions, log_probs, values

    def evaluate_actions(self, observations: torch.Tensor, stored_actions: torch.Tensor):
        values = self.value(observations)
        if self.is_discrete:
            distribution = Categorical(logits=self.actor(observations))
            actions = stored_actions.long().reshape(-1)
            return distribution.log_prob(actions), distribution.entropy(), values

        distribution = self._continuous_distribution(observations)
        log_probs = (
            distribution.log_prob(stored_actions).sum(-1)
            - self._tanh_log_det(stored_actions)
            - torch.log(self.action_scale).sum()
        )
        # Base Gaussian entropy is a stable exploration diagnostic. The tanh
        # transform has no simple closed-form entropy and entropy_coef defaults to 0.
        entropy = distribution.entropy().sum(-1)
        return log_probs, entropy, values

