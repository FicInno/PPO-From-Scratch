import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.make_env import make_env
from models.actor_critic import ActorCritic
from utils.evaluation import evaluate_policy
from utils.normalization import ObservationNormalizer
from utils.training import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    config = load_config(args.run_dir / "config.yaml")
    env = make_env(config["env_id"], args.seed)
    model = ActorCritic(
        int(np.prod(env.observation_space.shape)),
        env.action_space,
        hidden_sizes=config.get("hidden_sizes", [64, 64]),
        log_std_init=float(config.get("log_std_init", -0.5)),
    )
    env.close()
    model.load_state_dict(torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    normalizer_path = args.run_dir / "observation_normalizer.npz"
    observation_normalizer = ObservationNormalizer.load(normalizer_path) if normalizer_path.exists() else None
    returns = evaluate_policy(
        model,
        config["env_id"],
        args.episodes,
        args.seed,
        torch.device("cpu"),
        observation_normalizer,
    )
    print(json.dumps({"returns": returns, "mean": float(np.mean(returns)), "std": float(np.std(returns))}, indent=2))


if __name__ == "__main__":
    main()

