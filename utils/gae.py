"""Generalized Advantage Estimation."""

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_ends: np.ndarray | None,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with correct handling of time-limit truncation.

    ``terminated`` is 1 only for a true terminal state. A truncated transition
    bootstraps from ``next_values`` but ends the recursive advantage trajectory.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    if episode_ends is None:
        episode_ends = terminated
    for index in reversed(range(len(rewards))):
        bootstrap_mask = 1.0 - float(terminated[index])
        continuation_mask = 1.0 - float(episode_ends[index])
        delta = rewards[index] + gamma * next_values[index] * bootstrap_mask - values[index]
        last_advantage = delta + gamma * gae_lambda * continuation_mask * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)

