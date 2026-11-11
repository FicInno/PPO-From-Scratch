"""Run the matched-seed PPO-Clip and PPO-Penalty comparison."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.training import load_config, train


def run_one(config: dict, seed: int, run_dir: Path, force: bool) -> None:
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"SKIP completed: {run_dir}")
        return
    print(f"RUN {run_dir} seed={seed}")
    summary = train(config, seed, run_dir)
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    clip_config = load_config(ROOT / "configs" / "ppo_clip.yaml")
    penalty_config = load_config(ROOT / "configs" / "ppo_penalty.yaml")
    if args.total_steps is not None:
        clip_config["total_steps"] = args.total_steps
        penalty_config["total_steps"] = args.total_steps

    for algorithm, config in (("ppo_clip", clip_config), ("ppo_penalty", penalty_config)):
        for seed in args.seeds:
            run_one(
                config.copy(),
                seed,
                ROOT / "results" / "logs" / "validated_main" / algorithm / f"seed_{seed}",
                args.force,
            )

if __name__ == "__main__":
    main()

