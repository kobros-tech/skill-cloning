"""Skill-isolation invariant check for Issue #3 / PR #5.

This is NOT a statistical retention experiment. Under the current independent-
skill architecture, a previously acquired Skill's network is not optimized
again during later acquisitions. Each retention re-evaluation uses the same
frozen network and the same deterministic skill-specific evaluation batch.
Therefore a zero retention delta is expected by construction.

What this check legitimately verifies is the implementation invariant: later
acquisition must not mutate a supposedly frozen skill. The check is useful as
a regression guard against accidental parameter leakage, but it is not
empirical evidence that catastrophic forgetting is absent in architectures
where interference is possible. A genuine empirical forgetting experiment
would require an interference-risking baseline such as a shared-parameter
architecture.
"""
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
# Retained as a practical sanity bound. A violation indicates an isolation
# implementation bug rather than an inferentially surprising observation.
RETENTION_TOLERANCE = 0.05

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


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize invariant checks without inferential statistics."""
    retention = raw[raw["is_retention_check"]].copy()
    rows = []
    group_cols = ["sequence", "stage", "new_task", "evaluated_skill"]
    for group_key, group in retention.groupby(group_cols, sort=False):
        values = group["retention_delta"].to_numpy(dtype=float)
        rows.append({
            "sequence": group_key[0],
            "stage": group_key[1],
            "new_task": group_key[2],
            "evaluated_skill": group_key[3],
            "n_runs": len(group),
            "mean_pre_accuracy": group["pre_accuracy"].mean(),
            "mean_post_accuracy": group["post_accuracy"].mean(),
            "mean_retention_delta": group["retention_delta"].mean(),
            "max_absolute_retention_delta": float(np.max(np.abs(values))) if len(values) else np.nan,
            "all_deltas_exactly_zero": bool(np.all(values == 0.0)) if len(values) else None,
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
    pre = summary["mean_pre_accuracy"].to_numpy()
    post = summary["mean_post_accuracy"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(summary))
    ax.scatter(x, pre, marker="o", label="pre (baseline, at acquisition)", zorder=3)
    ax.scatter(x, post, marker="x", label="post (re-evaluated after later acquisitions)", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Frozen-skill accuracy on its stable eval set")
    ax.set_title(
        "Skill-isolation invariant check: frozen-skill accuracy before vs.\n"
        "after later acquisitions (identical points confirm isolation, not a statistical result)"
    )
    ax.legend(fontsize=8)
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

    print("Skill-isolation invariant check (not a statistical retention experiment)")
    print("Retention delta is expected to be exactly zero because stored skills are frozen")
    print(f"Seeds per sequence: {N_SEEDS}; sequences: {len(SEQUENCES)}")
    print(f"Sanity tolerance: {RETENTION_TOLERANCE:.1%} absolute accuracy drop")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
