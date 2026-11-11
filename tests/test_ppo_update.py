import gymnasium as gym
import torch

from algorithms.ppo_clip import PPOClip
from models.actor_critic import ActorCritic
from utils.buffer import RolloutBatch


def test_ppo_clip_update_is_finite_and_changes_policy():
    env = gym.make("Pendulum-v1")
    try:
        model = ActorCritic(3, env.action_space)
        observations = torch.randn(32, 3)
        with torch.no_grad():
            _, actions, old_log_probs, values = model.act(observations)
        batch = RolloutBatch(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=torch.linspace(-1.0, 1.0, 32),
            returns=values + torch.randn(32) * 0.1,
        )
        algorithm = PPOClip(model, 3e-4, 0.2, 0.5, 0.0, 0.5)
        before = [parameter.detach().clone() for parameter in model.policy_parameters()]
        metrics = algorithm.update(batch, epochs=2, mini_batch_size=8)
        assert torch.isfinite(torch.tensor([metrics.policy_loss, metrics.value_loss, metrics.approximate_kl])).all()
        assert any(not torch.equal(old, new) for old, new in zip(before, model.policy_parameters()))
    finally:
        env.close()
