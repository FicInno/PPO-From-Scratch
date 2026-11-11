import gymnasium as gym
import torch

from models.actor_critic import ActorCritic


def test_continuous_policy_actions_are_bounded_and_log_probs_repeatable():
    env = gym.make("Pendulum-v1")
    model = ActorCritic(3, env.action_space)
    observations = torch.randn(32, 3)
    env_actions, raw_actions, log_probs, _ = model.act(observations)
    repeated, entropy, values = model.evaluate_actions(observations, raw_actions)
    base = model._continuous_distribution(observations)
    expected = (
        base.log_prob(raw_actions).sum(-1)
        - model._tanh_log_det(raw_actions)
        - torch.log(model.action_scale).sum()
    )
    assert torch.all(env_actions <= 2.0)
    assert torch.all(env_actions >= -2.0)
    assert torch.allclose(log_probs, repeated, atol=1e-5)
    assert torch.allclose(log_probs, expected, atol=1e-5)
    assert entropy.shape == (32,)
    assert values.shape == (32,)
    env.close()


def test_discrete_policy_shapes():
    env = gym.make("CartPole-v1")
    model = ActorCritic(4, env.action_space)
    observations = torch.randn(8, 4)
    actions, stored, log_probs, values = model.act(observations)
    assert actions.shape == stored.shape == log_probs.shape == values.shape == (8,)
    env.close()

