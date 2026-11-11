"""On-policy rollout buffer."""

from dataclasses import dataclass

import numpy as np
import torch

from .gae import compute_gae


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, discrete: bool) -> None:
        self.capacity = capacity
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        action_shape = (capacity,) if discrete else (capacity, action_dim)
        action_dtype = np.int64 if discrete else np.float32
        self.actions = np.zeros(action_shape, dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.episode_ends = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, observation, action, reward, terminated, episode_end, value, next_value, log_prob) -> None:
        if self.position >= self.capacity:
            raise IndexError("rollout buffer is full")
        index = self.position
        self.observations[index] = np.asarray(observation, dtype=np.float32).reshape(-1)
        self.actions[index] = np.asarray(action).reshape(self.actions[index].shape)
        self.rewards[index] = reward
        self.terminated[index] = terminated
        self.episode_ends[index] = episode_end
        self.values[index] = value
        self.next_values[index] = next_value
        self.log_probs[index] = log_prob
        self.position += 1

    def finish(self, gamma: float, gae_lambda: float, normalize_advantage: bool, device: torch.device) -> RolloutBatch:
        count = self.position
        advantages, returns = compute_gae(
            self.rewards[:count],
            self.values[:count],
            self.next_values[:count],
            self.terminated[:count],
            self.episode_ends[:count],
            gamma,
            gae_lambda,
        )
        if normalize_advantage and count > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return RolloutBatch(
            observations=torch.as_tensor(self.observations[:count], device=device),
            actions=torch.as_tensor(self.actions[:count], device=device),
            old_log_probs=torch.as_tensor(self.log_probs[:count], device=device),
            advantages=torch.as_tensor(advantages, device=device),
            returns=torch.as_tensor(returns, device=device),
        )


