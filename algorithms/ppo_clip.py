"""PPO with the clipped surrogate objective."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from utils.buffer import RolloutBatch


@dataclass
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    kl_coefficient: float


class PPOClip:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        clip_epsilon: float,
        value_coefficient: float,
        entropy_coefficient: float,
        max_grad_norm: float,
    ) -> None:
        self.model = model
        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    def update(self, batch: RolloutBatch, epochs: int, mini_batch_size: int) -> UpdateMetrics:
        collected: list[tuple[float, ...]] = []
        size = batch.observations.shape[0]
        for _ in range(epochs):
            for indices in torch.randperm(size, device=batch.observations.device).split(mini_batch_size):
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    batch.observations[indices], batch.actions[indices]
                )
                log_ratio = new_log_probs - batch.old_log_probs[indices]
                ratio = log_ratio.exp()
                advantages = batch.advantages[indices]
                objective = torch.minimum(
                    ratio * advantages,
                    ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages,
                )
                policy_loss = -objective.mean()
                value_loss = nn.functional.mse_loss(values, batch.returns[indices])
                entropy_mean = entropy.mean()
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # The actor and critic are separate networks. Clipping them as one
                # parameter group lets the much larger value gradient suppress the
                # policy update on reward scales such as Pendulum's.
                nn.utils.clip_grad_norm_(self.model.policy_parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.value_parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                collected.append(
                    (policy_loss.item(), value_loss.item(), entropy_mean.item(), approximate_kl.item(), clip_fraction.item())
                )
        means = np.asarray(collected, dtype=np.float64).mean(axis=0)
        return UpdateMetrics(*means.tolist(), kl_coefficient=float("nan"))

