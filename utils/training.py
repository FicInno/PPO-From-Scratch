"""End-to-end on-policy PPO training loop and experiment logging."""

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from algorithms.ppo_clip import PPOClip
from algorithms.ppo_penalty import PPOPenalty
from environments.make_env import make_env
from models.actor_critic import ActorCritic
from utils.buffer import RolloutBuffer
from utils.evaluation import evaluate_policy
from utils.normalization import ObservationNormalizer
from utils.seed import set_global_seed


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "device": "cpu",
    }


def train(config: dict[str, Any], seed: int, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_metadata(), handle, ensure_ascii=False, indent=2)

    set_global_seed(seed)
    device = torch.device("cpu")
    env_id = str(config["env_id"])
    env = make_env(env_id, seed)
    observation, _ = env.reset(seed=seed)
    observation_dim = int(np.prod(env.observation_space.shape))
    discrete = hasattr(env.action_space, "n")
    action_dim = 1 if discrete else int(np.prod(env.action_space.shape))
    model = ActorCritic(
        observation_dim,
        env.action_space,
        hidden_sizes=config.get("hidden_sizes", [64, 64]),
        log_std_init=float(config.get("log_std_init", -0.5)),
    ).to(device)
    observation_normalizer = None
    if bool(config.get("normalize_observation", False)):
        observation_normalizer = ObservationNormalizer((observation_dim,))
        observation_normalizer.update(observation)

    common = dict(
        model=model,
        learning_rate=float(config["learning_rate"]),
        value_coefficient=float(config["value_coefficient"]),
        entropy_coefficient=float(config["entropy_coefficient"]),
        max_grad_norm=float(config.get("max_grad_norm", 0.5)),
    )
    algorithm_name = str(config["algorithm"])
    if algorithm_name == "ppo_clip":
        algorithm = PPOClip(clip_epsilon=float(config["clip_epsilon"]), **common)
    elif algorithm_name == "ppo_penalty":
        algorithm = PPOPenalty(
            initial_beta=float(config["initial_beta"]),
            target_kl=float(config["target_kl"]),
            adaptation_factor=float(config["adaptation_factor"]),
            min_beta=float(config.get("min_beta", 1e-3)),
            max_beta=float(config.get("max_beta", 1e3)),
            **common,
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm_name}")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    except ImportError:
        pass

    total_steps = int(config["total_steps"])
    rollout_steps = int(config["rollout_steps"])
    reward_scale = float(config.get("reward_scale", 1.0))
    total_updates = math.ceil(total_steps / rollout_steps)
    evaluation_interval = int(config.get("evaluation_interval", 5))
    evaluation_episodes = int(config.get("evaluation_episodes", 5))
    final_evaluation_episodes = int(config.get("final_evaluation_episodes", 10))
    progress_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    steps_done = 0
    update_number = 0
    episode_number = 0
    current_return = 0.0
    current_length = 0
    start_time = time.perf_counter()

    try:
        while steps_done < total_steps:
            current_rollout = min(rollout_steps, total_steps - steps_done)
            buffer = RolloutBuffer(current_rollout, observation_dim, action_dim, discrete)
            for _ in range(current_rollout):
                model_observation = (
                    observation_normalizer.normalize(observation)
                    if observation_normalizer is not None
                    else observation
                )
                obs_tensor = torch.as_tensor(model_observation, dtype=torch.float32, device=device).reshape(1, -1)
                with torch.no_grad():
                    env_action, stored_action, log_prob, value = model.act(obs_tensor)
                if discrete:
                    action_for_env = int(env_action.item())
                    action_for_buffer = int(stored_action.item())
                else:
                    action_for_env = env_action.cpu().numpy().reshape(env.action_space.shape)
                    action_for_buffer = stored_action.cpu().numpy().reshape(-1)

                next_observation, reward, terminated, truncated, _ = env.step(action_for_env)
                episode_end = bool(terminated or truncated)
                if observation_normalizer is not None:
                    observation_normalizer.update(next_observation)
                    model_next_observation = observation_normalizer.normalize(next_observation)
                else:
                    model_next_observation = next_observation
                with torch.no_grad():
                    next_tensor = torch.as_tensor(
                        model_next_observation, dtype=torch.float32, device=device
                    ).reshape(1, -1)
                    next_value = float(model.value(next_tensor).item())
                buffer.add(
                    model_observation,
                    action_for_buffer,
                    float(reward) * reward_scale,
                    bool(terminated),
                    episode_end,
                    float(value.item()),
                    next_value,
                    float(log_prob.item()),
                )
                steps_done += 1
                current_return += float(reward)
                current_length += 1
                observation = next_observation
                if episode_end:
                    episode_rows.append(
                        {
                            "step": steps_done,
                            "episode": episode_number,
                            "return": current_return,
                            "length": current_length,
                        }
                    )
                    episode_number += 1
                    current_return = 0.0
                    current_length = 0
                    observation, _ = env.reset()
                    if observation_normalizer is not None:
                        observation_normalizer.update(observation)

            batch = buffer.finish(
                gamma=float(config["gamma"]),
                gae_lambda=float(config["gae_lambda"]),
                normalize_advantage=bool(config.get("normalize_advantage", True)),
                device=device,
            )
            if bool(config.get("anneal_learning_rate", False)):
                remaining_fraction = 1.0 - update_number / total_updates
                current_learning_rate = float(config["learning_rate"]) * remaining_fraction
                for parameter_group in algorithm.optimizer.param_groups:
                    parameter_group["lr"] = current_learning_rate
            else:
                current_learning_rate = float(config["learning_rate"])
            metrics = algorithm.update(
                batch,
                epochs=int(config["ppo_epochs"]),
                mini_batch_size=int(config["mini_batch_size"]),
            )
            update_number += 1
            if update_number % evaluation_interval == 0 or steps_done == total_steps:
                evaluation_returns = evaluate_policy(
                    model,
                    env_id,
                    evaluation_episodes,
                    # Fixed evaluation episodes make learning-curve points and
                    # matched-seed algorithm comparisons directly comparable.
                    seed + 100_000,
                    device,
                    observation_normalizer,
                )
                evaluation_mean = float(np.mean(evaluation_returns))
                evaluation_std = float(np.std(evaluation_returns))
            else:
                evaluation_mean = float("nan")
                evaluation_std = float("nan")

            row = {
                "step": steps_done,
                "update": update_number,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "entropy": metrics.entropy,
                "approximate_kl": metrics.approximate_kl,
                "clip_fraction": metrics.clip_fraction,
                "kl_coefficient": metrics.kl_coefficient,
                "learning_rate": current_learning_rate,
                "evaluation_return_mean": evaluation_mean,
                "evaluation_return_std": evaluation_std,
            }
            progress_rows.append(row)
            if writer is not None:
                for key, value in row.items():
                    if key not in {"step", "update"} and np.isfinite(value):
                        writer.add_scalar(key, value, steps_done)
            if update_number % 5 == 0 or steps_done == total_steps:
                _write_csv(output_dir / "progress.csv", progress_rows)
                _write_csv(output_dir / "episodes.csv", episode_rows)

        final_returns = evaluate_policy(
            model,
            env_id,
            final_evaluation_episodes,
            seed + 900_000,
            device,
            observation_normalizer,
        )
        elapsed_seconds = time.perf_counter() - start_time
        summary = {
            "algorithm": algorithm_name,
            "env_id": env_id,
            "seed": seed,
            "total_steps": total_steps,
            "gae_lambda": float(config["gae_lambda"]),
            "normalize_advantage": bool(config.get("normalize_advantage", True)),
            "normalize_observation": observation_normalizer is not None,
            "reward_scale": reward_scale,
            "final_evaluation_returns": final_returns,
            "final_evaluation_mean": float(np.mean(final_returns)),
            "final_evaluation_std": float(np.std(final_returns)),
            "training_episode_return_mean_last_10": float(np.mean([r["return"] for r in episode_rows[-10:]])),
            "elapsed_seconds": elapsed_seconds,
        }
        torch.save(model.state_dict(), output_dir / "model.pt")
        if observation_normalizer is not None:
            observation_normalizer.save(output_dir / "observation_normalizer.npz")
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return summary
    finally:
        env.close()
        if writer is not None:
            writer.close()

