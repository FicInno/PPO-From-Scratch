"""Analyze the five-seed PPO experiment against a random-policy baseline."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dirs(parent: Path) -> list[Path]:
    return sorted(path.parent for path in parent.glob("seed_*/summary.json"))


def finite_curve(directory: Path, field: str) -> list[tuple[int, float]]:
    rows = load_csv(directory / "progress.csv")
    return [
        (int(row["step"]), row[field])
        for row in rows
        if field in row and math.isfinite(row[field])
    ]


def curve_matrix(directories: list[Path], field: str) -> tuple[np.ndarray, np.ndarray]:
    curves = [finite_curve(directory, field) for directory in directories]
    common_steps = sorted(set.intersection(*(set(step for step, _ in curve) for curve in curves)))
    matrix = np.asarray([[dict(curve)[step] for step in common_steps] for curve in curves])
    return np.asarray(common_steps), matrix


def bootstrap_ci(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def aggregate_group(label: str, directories: list[Path]) -> tuple[dict, list[dict]]:
    summaries = [load_json(directory / "summary.json") for directory in directories]
    final_returns = np.asarray([item["final_evaluation_mean"] for item in summaries])
    last_three, aucs = [], []
    for directory in directories:
        curve = finite_curve(directory, "evaluation_return_mean")
        steps = np.asarray([point[0] for point in curve])
        values = np.asarray([point[1] for point in curve])
        last_three.append(float(values[-3:].mean()))
        aucs.append(float(np.trapezoid(values, steps) / (steps[-1] - steps[0])))
    ci_low, ci_high = bootstrap_ci(final_returns)
    aggregate = {
        "group": label,
        "seeds": len(summaries),
        "mean_final_return": float(final_returns.mean()),
        "std_across_seeds": float(final_returns.std(ddof=1)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "mean_last_3_evaluations": float(np.mean(last_three)),
        "mean_learning_curve_auc": float(np.mean(aucs)),
        "mean_runtime_seconds": float(np.mean([item["elapsed_seconds"] for item in summaries])),
    }
    per_seed = [
        {
            "group": label,
            "seed": item["seed"],
            "final_evaluation_mean": item["final_evaluation_mean"],
            "final_evaluation_std": item["final_evaluation_std"],
            "last_3_evaluations": last_three[index],
            "learning_curve_auc": aucs[index],
            "runtime_seconds": item["elapsed_seconds"],
        }
        for index, item in enumerate(summaries)
    ]
    return aggregate, per_seed


def random_baseline_row() -> dict:
    directories = run_dirs(ROOT / "results" / "reference" / "random")
    if not directories:
        raise RuntimeError("Random-policy baseline is incomplete")
    values = np.asarray([load_json(path / "summary.json")["final_evaluation_mean"] for path in directories])
    low, high = bootstrap_ci(values)
    return {
        "group": "Random",
        "seeds": len(values),
        "mean_final_return": float(values.mean()),
        "std_across_seeds": float(values.std(ddof=1)),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    experiment_root = ROOT / "results" / "logs" / "validated_main"
    groups = {
        "PPO-Clip": run_dirs(experiment_root / "ppo_clip"),
        "PPO-Penalty": run_dirs(experiment_root / "ppo_penalty"),
    }
    if any(len(paths) < 2 for paths in groups.values()):
        raise RuntimeError("Experiment is incomplete; at least two seeds per group are required")

    summary_rows, per_seed_rows = [], []
    for label, directories in groups.items():
        aggregate, per_seed = aggregate_group(label, directories)
        summary_rows.append(aggregate)
        per_seed_rows.extend(per_seed)
    save_csv(TABLES / "validated_summary.csv", summary_rows)
    save_csv(TABLES / "validated_per_seed.csv", per_seed_rows)

    clip_by_seed = {row["seed"]: row for row in per_seed_rows if row["group"] == "PPO-Clip"}
    penalty_by_seed = {row["seed"]: row for row in per_seed_rows if row["group"] == "PPO-Penalty"}
    matched = sorted(set(clip_by_seed) & set(penalty_by_seed))
    differences = np.asarray(
        [clip_by_seed[seed]["final_evaluation_mean"] - penalty_by_seed[seed]["final_evaluation_mean"] for seed in matched]
    )
    difference_low, difference_high = bootstrap_ci(differences)
    paired_row = {
        "comparison": "PPO-Clip minus PPO-Penalty",
        "matched_seeds": len(matched),
        "mean_paired_difference": float(differences.mean()),
        "std_paired_difference": float(differences.std(ddof=1)),
        "bootstrap_ci_low": difference_low,
        "bootstrap_ci_high": difference_high,
    }
    save_csv(TABLES / "validated_paired_comparison.csv", [paired_row])

    random_baseline = random_baseline_row()
    save_csv(TABLES / "random_baseline_summary.csv", [random_baseline])

    colors = {"PPO-Clip": "#2563eb", "PPO-Penalty": "#dc2626"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, directories in groups.items():
        steps, matrix = curve_matrix(directories, "evaluation_return_mean")
        mean = matrix.mean(axis=0)
        ci = 2.776 * matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
        ax.plot(steps, mean, label=label, color=colors[label], linewidth=2)
        ax.fill_between(steps, mean - ci, mean + ci, color=colors[label], alpha=0.16)
    ax.set(
        xlabel="Environment steps",
        ylabel="Deterministic evaluation return",
        title="Five-seed PPO comparison (mean and 95% t-interval)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "validated_learning_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = (
        ("approximate_kl", "Approximate KL"),
        ("clip_fraction", "Clip fraction"),
        ("kl_coefficient", "KL coefficient"),
    )
    for ax, (field, title) in zip(axes, specs):
        for label, directories in groups.items():
            try:
                steps, matrix = curve_matrix(directories, field)
            except (KeyError, ValueError):
                continue
            if matrix.size:
                ax.plot(steps, matrix.mean(axis=0), label=label, color=colors[label])
        ax.set_title(title)
        ax.set_xlabel("Steps")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "validated_diagnostics.png", dpi=180)
    plt.close(fig)

    comparison_rows = summary_rows + [random_baseline]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = [row["group"] for row in comparison_rows]
    means = [row["mean_final_return"] for row in comparison_rows]
    lows = [mean - row["bootstrap_ci_low"] for mean, row in zip(means, comparison_rows)]
    highs = [row["bootstrap_ci_high"] - mean for mean, row in zip(means, comparison_rows)]
    ax.bar(names, means, color=["#2563eb", "#dc2626", "#6b7280"])
    ax.errorbar(np.arange(len(names)), means, yerr=[lows, highs], fmt="none", color="black", capsize=5)
    ax.set(ylabel="Final evaluation return", title="Final performance relative to random policy")
    fig.tight_layout()
    fig.savefig(FIGURES / "validated_final_performance.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"groups": summary_rows, "paired": paired_row, "random": random_baseline}, indent=2))


if __name__ == "__main__":
    main()
