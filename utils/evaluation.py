import torch

from environments.make_env import make_env


@torch.no_grad()
def evaluate_policy(
    model,
    env_id: str,
    episodes: int,
    seed: int,
    device: torch.device,
    observation_normalizer=None,
) -> list[float]:
    env = make_env(env_id, seed)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                model_observation = (
                    observation_normalizer.normalize(observation)
                    if observation_normalizer is not None
                    else observation
                )
                tensor = torch.as_tensor(model_observation, dtype=torch.float32, device=device).reshape(1, -1)
                env_action, _, _, _ = model.act(tensor, deterministic=True)
                if model.is_discrete:
                    action = int(env_action.item())
                else:
                    action = env_action.cpu().numpy().reshape(env.action_space.shape)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                done = terminated or truncated
            returns.append(episode_return)
    finally:
        env.close()
    return returns

