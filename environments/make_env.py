import gymnasium as gym


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

