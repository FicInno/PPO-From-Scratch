"""Evaluate a random policy as the absolute performance baseline."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def evaluate_random(env_id: str, episodes: int, seed: int) -> list[float]:
    env = gym.make(env_id)
    env.action_space.seed(seed)
    returns = []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            done = False
            total = 0.0
            while not done:
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
    finally:
        env.close()
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    for seed in args.seeds:
        run_dir = ROOT / "results" / "reference" / "random" / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            print(f"SKIP completed: {run_dir}")
            continue
        returns = evaluate_random(args.env_id, args.episodes, seed + 900_000)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "Random",
            "seed": seed,
            "final_evaluation_returns": returns,
            "final_evaluation_mean": float(np.mean(returns)),
            "final_evaluation_std": float(np.std(returns)),
            "elapsed_seconds": 0.0,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"seed={seed} mean={np.mean(returns):.2f}")


if __name__ == "__main__":
    main()
