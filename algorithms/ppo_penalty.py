"""PPO with an adaptive KL-penalized surrogate objective."""

import numpy as np
import torch
from torch import nn

from algorithms.ppo_clip import UpdateMetrics
from utils.buffer import RolloutBatch


class PPOPenalty:
    """KL-penalized PPO.

    The sampled non-negative ``(ratio - 1) - log(ratio)`` estimator is used for
    the old-to-new KL penalty. Beta is adapted after each rollout update.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        initial_beta: float,
        target_kl: float,
        adaptation_factor: float,
        min_beta: float,
        max_beta: float,
        value_coefficient: float,
        entropy_coefficient: float,
        max_grad_norm: float,
    ) -> None:
        self.model = model
        self.beta = initial_beta
        self.target_kl = target_kl
        self.adaptation_factor = adaptation_factor
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    @torch.no_grad()
    def measure_full_batch_kl(self, batch: RolloutBatch) -> float:
        """Measure old-to-new KL on the full rollout after all PPO epochs."""
        new_log_probs, _, _ = self.model.evaluate_actions(batch.observations, batch.actions)
        log_ratio = new_log_probs - batch.old_log_probs
        ratio = log_ratio.exp()
        return float(((ratio - 1.0) - log_ratio).mean().item())

    def update(self, batch: RolloutBatch, epochs: int, mini_batch_size: int) -> UpdateMetrics:
        collected: list[tuple[float, float, float]] = []
        size = batch.observations.shape[0]
        beta_used = self.beta
        for _ in range(epochs):
            for indices in torch.randperm(size, device=batch.observations.device).split(mini_batch_size):
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    batch.observations[indices], batch.actions[indices]
                )
                log_ratio = new_log_probs - batch.old_log_probs[indices]
                ratio = log_ratio.exp()
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                surrogate = (ratio * batch.advantages[indices]).mean()
                policy_loss = -surrogate + beta_used * approximate_kl
                value_loss = nn.functional.mse_loss(values, batch.returns[indices])
                entropy_mean = entropy.mean()
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.policy_parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.value_parameters(), self.max_grad_norm)
                self.optimizer.step()
                collected.append((policy_loss.item(), value_loss.item(), entropy_mean.item()))

        means = np.asarray(collected, dtype=np.float64).mean(axis=0)
        measured_kl = self.measure_full_batch_kl(batch)
        if measured_kl > 1.5 * self.target_kl:
            self.beta = min(self.beta * self.adaptation_factor, self.max_beta)
        elif measured_kl < self.target_kl / 1.5:
            self.beta = max(self.beta / self.adaptation_factor, self.min_beta)
        return UpdateMetrics(
            policy_loss=float(means[0]),
            value_loss=float(means[1]),
            entropy=float(means[2]),
            approximate_kl=measured_kl,
            clip_fraction=float("nan"),
            kl_coefficient=beta_used,
        )

