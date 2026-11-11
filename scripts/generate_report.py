"""Generate the five-seed experiment report from result tables."""

import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DATE = "2026-11-11"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / "tables" / name).open("r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered = []
        for key, _ in columns:
            value = row[key]
            try:
                number = float(value)
                value = str(int(number)) if key in {"seed", "seeds", "matched_seeds"} else f"{number:.2f}"
            except ValueError:
                pass
            rendered.append(value)
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    summary = read_csv("validated_summary.csv")
    per_seed = read_csv("validated_per_seed.csv")
    paired = read_csv("validated_paired_comparison.csv")[0]
    random_baseline = read_csv("random_baseline_summary.csv")

    sample_path = next((ROOT / "results" / "logs" / "validated_main").glob("*/*/summary.json"))
    run_info = json.loads(sample_path.read_text(encoding="utf-8"))
    metadata = json.loads((sample_path.parent / "metadata.json").read_text(encoding="utf-8"))
    with (ROOT / "configs" / "ppo_clip.yaml").open("r", encoding="utf-8") as handle:
        clip_config = yaml.safe_load(handle)
    with (ROOT / "configs" / "ppo_penalty.yaml").open("r", encoding="utf-8") as handle:
        penalty_config = yaml.safe_load(handle)

    main_table = table(
        summary,
        [
            ("group", "Method"),
            ("seeds", "Seeds"),
            ("mean_final_return", "Final Return"),
            ("std_across_seeds", "Across-Seed SD"),
            ("bootstrap_ci_low", "95% CI Low"),
            ("bootstrap_ci_high", "95% CI High"),
            ("mean_learning_curve_auc", "Learning-Curve AUC"),
        ],
    )
    seed_table = table(
        per_seed,
        [
            ("group", "Method"),
            ("seed", "Seed"),
            ("final_evaluation_mean", "Final Return"),
            ("last_3_evaluations", "Last 3 Evaluations"),
            ("learning_curve_auc", "Curve AUC"),
        ],
    )
    random_table = table(
        random_baseline,
        [
            ("group", "Baseline"),
            ("seeds", "Seeds"),
            ("mean_final_return", "Final Return"),
            ("std_across_seeds", "Across-Seed SD"),
        ],
    )

    report = f"""# PPO from Scratch: Five-Seed Clip and KL-Penalty Comparison

> PyTorch implementation and matched-seed experiment  
> Experiment date: {EXPERIMENT_DATE}  
> Environment: Pendulum-v1

## Abstract

This project implements PPO-Clip and adaptive KL-Penalty PPO in PyTorch without a ready-made RL algorithm. On Pendulum-v1, each method was trained for {run_info['total_steps']:,} environment steps with {summary[0]['seeds']} matched seeds. The final policies were evaluated deterministically on {clip_config['final_evaluation_episodes']} separately seeded episodes per run, alongside a random-policy baseline.

Mean final returns were {float(summary[0]['mean_final_return']):.2f} for PPO-Clip and {float(summary[1]['mean_final_return']):.2f} for PPO-Penalty, compared with {float(random_baseline[0]['mean_final_return']):.2f} for the random policy. PPO-Clip achieved the higher mean learning-curve AUC. The paired final-return difference was {float(paired['mean_paired_difference']):.2f}, with a 95% bootstrap interval of [{float(paired['bootstrap_ci_low']):.2f}, {float(paired['bootstrap_ci_high']):.2f}]; the final-return ranking remains unresolved in this experiment.

## 1. Implementation

- Separate actor and critic MLPs with independent gradient clipping.
- Categorical policies and tanh-squashed Gaussian policies with tanh and affine Jacobian corrections.
- Generalized Advantage Estimation with distinct true-termination and time-limit-truncation handling.
- Running observation normalization persisted with each trained policy.
- PPO-Clip and adaptive KL-Penalty updates, configurable learning-rate scheduling, raw-reward logging, and optional reward scaling during training.
- CSV/TensorBoard logging, saved final model weights, fixed-seed evaluation, matched-seed aggregation, bootstrap intervals, and a random-policy baseline.

## 2. Protocol

| Parameter | Value |
| --- | --- |
| Training seeds | {', '.join(str(index) for index in range(int(float(summary[0]['seeds']))))} |
| Environment steps per run | {run_info['total_steps']:,} |
| Rollout / mini-batch | {clip_config['rollout_steps']} / {clip_config['mini_batch_size']} |
| PPO epochs | {clip_config['ppo_epochs']} |
| Learning rate | {clip_config['learning_rate']} |
| Gamma / GAE lambda | {clip_config['gamma']} / {clip_config['gae_lambda']} |
| Observation normalization | {clip_config['normalize_observation']} |
| Reward scale for training targets | {clip_config['reward_scale']} |
| Final evaluation | {clip_config['final_evaluation_episodes']} deterministic episodes per seed |
| PPO-Penalty target KL / initial beta | {penalty_config['target_kl']} / {penalty_config['initial_beta']} |

The configurations saved with each run match those in `configs/`. Periodic evaluations reuse episode seeds within each run; final evaluations use a separate fixed seed range. Each run saves its final model weights in `model.pt`.

## 3. Results

### 3.1 Main comparison

{main_table}

![Five-seed learning curves](results/figures/validated_learning_curve.png)

![Optimization diagnostics](results/figures/validated_diagnostics.png)

The KL lines use different summaries: PPO-Clip averages estimates across update mini-batches, while PPO-Penalty reports a post-update estimate over the full rollout.

### 3.2 Per-seed results

{seed_table}

### 3.3 Random-policy baseline

{random_table}

![Final performance relative to random policy](results/figures/validated_final_performance.png)

## 4. Statistical interpretation

Across the five matched seeds, the mean final-return difference (PPO-Clip minus PPO-Penalty) was {float(paired['mean_paired_difference']):.2f}. A 20,000-resample paired bootstrap gives a 95% interval of [{float(paired['bootstrap_ci_low']):.2f}, {float(paired['bootstrap_ci_high']):.2f}]. Because it spans zero, the final-return data do not establish an advantage for either method.

PPO-Clip had the higher learning-curve AUC in all five matched seeds. AUC is the step-normalized trapezoidal area of periodic evaluation returns from the first evaluation point through 100,000 steps. The last-three-evaluation mean describes late-training performance; the final-return measure uses a separate set of episode seeds.

## 5. Correctness and result traceability

Automated tests cover numerical GAE behavior, truncation bootstrapping, rollout shapes, action bounds, transformed log probabilities, observation-normalizer persistence, discrete-policy interfaces, a sampled KL estimate, and a finite PPO update that changes policy parameters.

```powershell
python -m pytest -q
python scripts/run_experiments.py
python scripts/run_random_baseline.py
python scripts/analyze_results.py
python scripts/generate_report.py
```

Runtime metadata for this experiment: Python {metadata['python'].split()[0]}, PyTorch {metadata['torch']}, Gymnasium {metadata['gymnasium']}, NumPy {metadata['numpy']}, {metadata['platform']}.

## 6. Limitations

- Results describe Pendulum-v1 under the reported 100,000-step protocol and PPO-Penalty configuration (target KL 0.003, initial beta 0.1).
- Five training seeds leave small between-method final-return differences uncertain.

## 7. Deliverables

- Core implementation: `models/`, `algorithms/`, `utils/`
- Experiment entry points: `scripts/`
- Experiment configurations: `configs/`
- Tests: `tests/`
- Seed-level logs and final model weights: `results/logs/validated_main/`
- Random-policy baseline: `results/reference/random/`
- Tables and figures: `results/tables/`, `results/figures/`
"""
    (ROOT / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    print(ROOT / "EXPERIMENT_REPORT.md")


if __name__ == "__main__":
    main()
