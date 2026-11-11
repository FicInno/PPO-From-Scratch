import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.training import load_config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO from scratch")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--env-id")
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--normalize-advantage", choices=["true", "false"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    for key in ("total_steps", "env_id", "gae_lambda"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.normalize_advantage is not None:
        config["normalize_advantage"] = args.normalize_advantage == "true"
    summary = train(config, args.seed, args.output_dir)
    print(summary)

