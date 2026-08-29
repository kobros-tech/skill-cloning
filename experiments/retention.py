"""Sequential skill-retention experiment for Issue #3 / PR #5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import compatibility as comp  # noqa: E402
from skill import Skill, TinyMLP  # noqa: E402
from tasks import sample_task  # noqa: E402

N_SEEDS = 15
N_TRAIN = 200
N_EVAL = 300
HIDDEN_DIM = 32
LR = 0.02
MAX_EPOCHS = 1500
ACC_TOL = 0.5
ACC_TARGET = 0.85
# Predeclared practical tolerance: five percentage points of absolute accuracy.
RETENTION_TOLERANCE = 0.05
BOOTSTRAP_SAMPLES = 2000

TASK_SEED_INDEX = {
    "addition": 0,
    "subtraction": 1,
    "multiplication": 2,
    "powers": 3,
    "squares": 4,
    "division": 5,
}

SEQUENCES = [
    ("addition", "multiplication", "division"),
    ("addition", "multiplication", "squares"),
    ("addition", "multiplication", "powers"),
    ("addition", "subtraction", "division"),
]


def _task_seed(seed: int, sequence_index: int, stage: int, task: str, offset: int) -> int:
    return (
        seed * 1_000_000
        + sequence_index * 10_000
        + stage * 1_000
        + TASK_SEED_INDEX[task]
        + offset
    )


def train_to_accuracy(net, X, y):
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train_step(X, y, lr=LR)
        if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
            return epoch, True
    return MAX_EPOCHS, False


def _acquire_target(skills, target, seed, sequence_index, stage):
    controller_seed = _task_seed(seed, sequence_index, stage, target, offset=20_000)
    decision = comp.decide(skills, target, base_seed=controller_seed)

    train_seed = _task_seed(seed, sequence_index, stage, target, offset=30_000)
    X_train, y_train = sample_task(target, N_TRAIN, seed=train_seed)

    if decision["action"] == "reuse":
        net = skills[decision["parent"]].net
        steps, success = 0, True
        source_task = decision["parent"]
    elif decision["action"] == "clone":
        source_task = decision["parent"]
        net = skills[source_task].net.clone()
        net.reset_optimizer()
        steps, success = train_to_accuracy(net, X_train, y_train)
    else:
        source_task = None
        net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=train_seed)
        net.reset_optimizer()
        steps, success = train_to_accuracy(net, X_train, y_train)

    return (
        Skill(target, net, origin=decision["action"], parent=source_task),
        {
            "strategy": decision["action"],
            "source_task": source_task,
            "compatibility_score": float(decision["score"]),
            "solve_accuracy": float(decision["solve_accuracy"]),
            "adaptation_steps": steps,
            "acquisition_success": bool(success),
        },
    )


def _retention_accuracy(skill, task, seed, sequence_index, skill_stage):
    """Evaluate on a stable skill-specific retention set across later stages."""
    eval_seed = _task_seed(seed, sequence_index, skill_stage, task, offset=40_000)
    X, y = sample_task(task, N_EVAL, seed=eval_seed)
    return float(skill.net.accuracy(X, y, tol=ACC_TOL))


def run_sequence(sequence: tuple[str, ...], seed: int, sequence_index: int):
    """Run one sequential acquisition and paired retention checks."""
    skills = {}
    baseline_accuracy = {}
    skill_stage = {}
    rows = []

    for stage, target in enumerate(sequence):
        new_skill, acquisition = _acquire_target(
            skills, target, seed, sequence_index, stage
        )

        if acquisition["acquisition_success"]:
            skills[target] = new_skill
            skill_stage[target] = stage
            baseline_accuracy[target] = _retention_accuracy(
                skills[target], target, seed, sequence_index, stage
            )

        for old_task, old_skill in skills.items():
            post_accuracy = _retention_accuracy(
                old_skill, old_task, seed, sequence_index, skill_stage[old_task]
            )
            is_baseline = old_task == target and skill_stage[old_task] == stage
            pre_accuracy = post_accuracy if is_baseline else baseline_accuracy[old_task]
            delta = 0.0 if is_baseline else post_accuracy - pre_accuracy

            rows.append({
                "sequence": "→".join(sequence),
                "sequence_index": sequence_index,
                "seed": seed,
                "stage": stage,
                "new_task": target,
                "evaluated_skill": old_task,
                "skill_stage": skill_stage[old_task],
                "is_retention_check": not is_baseline,
                "strategy": acquisition["strategy"],
                "source_task": acquisition["source_task"],
                "compatibility_score": acquisition["compatibility_score"],
                "solve_accuracy": acquisition["solve_accuracy"],
                "adaptation_steps": acquisition["adaptation_steps"],
                "new_skill_acquisition_success": acquisition["acquisition_success"],
                "pre_accuracy": pre_accuracy,
                "post_accuracy": post_accuracy,
                "retention_delta": delta,
                "retention_ratio": (
                    post_accuracy / pre_accuracy if pre_accuracy > 0 else np.nan
                ),
                "retention_pass": delta >= -RETENTION_TOLERANCE,
            })

    return rows


def run_all_sequences(n_seeds: int = N_SEEDS):
    rows = []
    for sequence_index, sequence in enumerate(SEQUENCES):
        for seed in range(n_seeds):
            rows.extend(run_sequence(sequence, seed, sequence_index))
    return rows


def bootstrap_ci(values, seed: int):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize only post-acquisition retention checks, excluding baselines."""
    retention = raw[raw["is_retention_check"]].copy()
    rows = []
    group_cols = ["sequence", "stage", "new_task", "evaluated_skill"]
    for group_index, (group_key, group) in enumerate(retention.groupby(group_cols, sort=False)):
        values = group["retention_delta"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(values, seed=12_345 + group_index)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        effect = float(np.mean(values) / std) if std > 0 else np.nan
        rows.append({
            "sequence": group_key[0],
            "stage": group_key[1],
            "new_task": group_key[2],
            "evaluated_skill": group_key[3],
            "n_runs": len(group),
            "mean_pre_accuracy": group["pre_accuracy"].mean(),
            "mean_post_accuracy": group["post_accuracy"].mean(),
            "mean_retention_delta": group["retention_delta"].mean(),
            "std_retention_delta": std,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "paired_effect_size": effect,
            "retention_pass_rate": group["retention_pass"].mean(),
            "retention_tolerance": RETENTION_TOLERANCE,
            "new_skill_success_rate": group["new_skill_acquisition_success"].mean(),
        })
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{r.sequence}\nstage {r.stage}: {r.evaluated_skill}"
        for r in summary.itertuples()
    ]
    means = summary["mean_retention_delta"].to_numpy()
    lower = means - summary["bootstrap_ci_low"].to_numpy()
    upper = summary["bootstrap_ci_high"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(range(len(summary)), means, yerr=[lower, upper], fmt="o")
    ax.axhline(-RETENTION_TOLERANCE, linestyle="--", linewidth=1)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy change after new-skill acquisition")
    ax.set_title("Retention of previously acquired skills")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    raw = pd.DataFrame(run_all_sequences())
    summary = summarize(raw)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw.to_csv(out_dir / "retention.csv", index=False)
    summary.to_csv(out_dir / "retention_summary.csv", index=False)
    make_plot(summary, out_dir / "plot_retention.png")

    print("Sequential skill retention / catastrophic-forgetting experiment")
    print(f"Seeds per sequence: {N_SEEDS}; sequences: {len(SEQUENCES)}")
    print(f"Retention tolerance: {RETENTION_TOLERANCE:.1%} absolute accuracy drop")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
