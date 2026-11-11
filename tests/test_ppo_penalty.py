import gymnasium as gym
import torch

from algorithms.ppo_penalty import PPOPenalty
from models.actor_critic import ActorCritic
from utils.buffer import RolloutBatch


def test_full_batch_kl_matches_manual_estimator():
    env = gym.make("Pendulum-v1")
    model = ActorCritic(3, env.action_space)
    observations = torch.randn(32, 3)
    with torch.no_grad():
        _, raw_actions, old_log_probs, _ = model.act(observations)
    batch = RolloutBatch(
        observations=observations,
        actions=raw_actions,
        old_log_probs=old_log_probs - 0.1,
        advantages=torch.zeros(32),
        returns=torch.zeros(32),
    )
    algorithm = PPOPenalty(
        model=model,
        learning_rate=3e-4,
        initial_beta=1.0,
        target_kl=0.01,
        adaptation_factor=2.0,
        min_beta=1e-3,
        max_beta=1e3,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_grad_norm=0.5,
    )
    expected = torch.exp(torch.tensor(0.1)) - 1.0 - 0.1
    assert abs(algorithm.measure_full_batch_kl(batch) - expected.item()) < 1e-6
    env.close()
