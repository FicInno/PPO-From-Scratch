"""Create a side-by-side GIF of untrained and trained Pendulum policies."""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.actor_critic import ActorCritic
from utils.normalization import ObservationNormalizer
from utils.seed import set_global_seed
from utils.training import load_config


def build_model(config: dict, seed: int) -> ActorCritic:
    set_global_seed(seed)
    env = gym.make(config["env_id"])
    try:
        model = ActorCritic(
            int(np.prod(env.observation_space.shape)),
            env.action_space,
            hidden_sizes=config.get("hidden_sizes", [64, 64]),
            log_std_init=float(config.get("log_std_init", -0.5)),
        )
    finally:
        env.close()
    model.eval()
    return model


@torch.no_grad()
def rollout(
    model: ActorCritic,
    env_id: str,
    seed: int,
    observation_normalizer: ObservationNormalizer | None = None,
) -> tuple[list[np.ndarray], float]:
    env = gym.make(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    episode_return = 0.0
    try:
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            frames.append(env.render())
            model_observation = (
                observation_normalizer.normalize(observation)
                if observation_normalizer is not None
                else observation
            )
            tensor = torch.as_tensor(model_observation, dtype=torch.float32).reshape(1, -1)
            action, _, _, _ = model.act(tensor, deterministic=True)
            if model.is_discrete:
                env_action = int(action.item())
            else:
                env_action = action.numpy().reshape(env.action_space.shape)
            observation, reward, terminated, truncated, _ = env.step(env_action)
            episode_return += float(reward)
            done = terminated or truncated
        frames.append(env.render())
    finally:
        env.close()
    return frames, episode_return


def make_comparison_gif(
    untrained_frames: list[np.ndarray],
    trained_frames: list[np.ndarray],
    untrained_return: float,
    trained_return: float,
    output_path: Path,
) -> None:
    count = min(len(untrained_frames), len(trained_frames))
    rendered: list[Image.Image] = []
    for index in range(0, count, 2):
        left = Image.fromarray(untrained_frames[index]).convert("RGB")
        right = Image.fromarray(trained_frames[index]).convert("RGB")
        width, height = left.size
        canvas = Image.new("RGB", (width * 2, height + 34), "white")
        canvas.paste(left, (0, 34))
        canvas.paste(right, (width, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"Untrained policy | Return: {untrained_return:.1f}", fill="black")
        draw.text((width + 12, 10), f"Trained PPO-Clip | Return: {trained_return:.1f}", fill="black")
        rendered.append(canvas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(
        output_path,
        save_all=True,
        append_images=rendered[1:],
        duration=70,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results" / "logs" / "validated_main" / "ppo_clip" / "seed_1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "media" / "pendulum_before_after.gif",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config = load_config(args.run_dir / "config.yaml")
    run_summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    training_seed = int(run_summary["seed"])
    untrained = build_model(config, seed=training_seed)
    trained = build_model(config, seed=training_seed)
    trained.load_state_dict(torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True))
    untrained_frames, untrained_return = rollout(untrained, config["env_id"], args.seed)
    normalizer_path = args.run_dir / "observation_normalizer.npz"
    observation_normalizer = ObservationNormalizer.load(normalizer_path) if normalizer_path.exists() else None
    trained_frames, trained_return = rollout(
        trained, config["env_id"], args.seed, observation_normalizer
    )
    make_comparison_gif(
        untrained_frames,
        trained_frames,
        untrained_return,
        trained_return,
        args.output,
    )
    print(f"Saved {args.output}")
    print(f"Untrained return: {untrained_return:.3f}")
    print(f"Trained return: {trained_return:.3f}")


if __name__ == "__main__":
    main()
