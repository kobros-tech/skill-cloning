"""Calibrate and audit the compatibility decision rule without using test data.

The original controller treated a high frozen-skill compatibility score as
sufficient evidence for ``reuse``. This experiment checks that assumption.
A skill is considered *actually solved* on a target when it reaches the same
85%-accuracy criterion used by the training experiments on an independent
calibration batch. Compatibility scores are evaluated on a separate probe
batch. No held-out test data are used for threshold selection.

The script trains one skill for each source task, then evaluates every
source -> target pair across the declared seeds. It reports both the original
score threshold and the independent calibration accuracy criterion, plus a
recommended score cutoff chosen on the calibration pairs to separate
same-task (solved) from cross-task (not-solved) cases.

Outputs:
    results/compatibility_calibration.csv
    results/compatibility_calibration_summary.csv
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

from compatibility import TAU_CLONE, TAU_SOLVE, compatibility_score  # noqa: E402
from skill import TinyMLP  # noqa: E402
from tasks import TASK_ORDER, sample_task  # noqa: E402

N_SEEDS = 15
N_TRAIN = 200
N_CALIBRATION = 300
HIDDEN_DIM = 32
LR = 0.02
MAX_EPOCHS = 1500
ACC_TOL = 0.5
ACC_TARGET = 0.85


def train_to_accuracy(net, X, y):
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train_step(X, y, lr=LR)
        if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
            return epoch
    return MAX_EPOCHS


def recommended_score_threshold(df: pd.DataFrame) -> float:
    """Choose a calibration-only score cutoff with best balanced accuracy.

    Positives are same-task evaluations, which are the only cases that should
    be eligible for zero-training reuse. Cross-task pairs are negatives. The
    threshold is selected only from this calibration data; test data are never
    involved.
    """
    scores = np.sort(df["compatibility_score"].unique())
    candidates = np.concatenate(([0.0], scores, [1.0]))
    best = None
    for threshold in candidates:
        pred = df["compatibility_score"] >= threshold
        actual = df["solved_label"]
        tp = np.sum(pred & actual)
        tn = np.sum(~pred & ~actual)
        positives = np.sum(actual)
        negatives = np.sum(~actual)
        balanced = 0.5 * (tp / positives + tn / negatives)
        false_positive = np.sum(pred & ~actual)
        key = (balanced, -false_positive, threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    return float(best[1])


def main():
    rows = []
    for seed in range(N_SEEDS):
        skills = {}
        for source_index, source_task in enumerate(TASK_ORDER):
            X, y = sample_task(source_task, N_TRAIN, seed=seed * 100 + source_index)
            net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + source_index)
            net.reset_optimizer()
            train_to_accuracy(net, X, y)
            skills[source_task] = net

        for source_task, net in skills.items():
            for target_index, target_task in enumerate(TASK_ORDER):
                # Independent calibration batch: distinct from both training
                # and the compatibility probe batch.
                X_cal, y_cal = sample_task(
                    target_task,
                    N_CALIBRATION,
                    seed=seed * 100 + 20_000 + target_index,
                )
                score = compatibility_score(net, target_task, seed * 100 + target_index)
                accuracy = net.accuracy(X_cal, y_cal, tol=ACC_TOL)
                mse = net.mse(X_cal, y_cal)
                solved = source_task == target_task and accuracy >= ACC_TARGET
                original_reuse = score >= TAU_SOLVE
                rows.append(
                    {
                        "seed": seed,
                        "source_task": source_task,
                        "target_task": target_task,
                        "compatibility_score": score,
                        "calibration_mse": mse,
                        "calibration_accuracy": accuracy,
                        "solved_label": solved,
                        "original_reuse": original_reuse,
                        "false_reuse": original_reuse and not solved,
                    }
                )

    raw = pd.DataFrame(rows)
    recommended = recommended_score_threshold(raw)
    raw["calibrated_score_reuse"] = raw["compatibility_score"] >= recommended
    raw["calibrated_false_reuse"] = raw["calibrated_score_reuse"] & ~raw["solved_label"]

    summary_rows = [
        {
            "metric": "original_tau_solve",
            "value": TAU_SOLVE,
            "false_reuse_count": int(raw["false_reuse"].sum()),
            "reuse_count": int(raw["original_reuse"].sum()),
        },
        {
            "metric": "recommended_tau_solve_calibration",
            "value": recommended,
            "false_reuse_count": int(raw["calibrated_false_reuse"].sum()),
            "reuse_count": int(raw["calibrated_score_reuse"].sum()),
        },
        {
            "metric": "tau_clone_reference",
            "value": TAU_CLONE,
            "false_reuse_count": np.nan,
            "reuse_count": np.nan,
        },
    ]
    summary = pd.DataFrame(summary_rows)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw.to_csv(out_dir / "compatibility_calibration.csv", index=False)
    summary.to_csv(out_dir / "compatibility_calibration_summary.csv", index=False)

    print("Compatibility decision-rule calibration")
    print(f"Seeds: {N_SEEDS}; tasks: {list(TASK_ORDER)}")
    print(f"Original tau_solve: {TAU_SOLVE:.6f}")
    print(f"Recommended calibration-only tau_solve: {recommended:.6f}")
    print(f"Original reuse decisions: {int(raw['original_reuse'].sum())}")
    print(f"Original false reuse decisions: {int(raw['false_reuse'].sum())}")
    print(f"Calibrated false reuse decisions: {int(raw['calibrated_false_reuse'].sum())}")
    print("\nOriginal false-reuse rate by source -> target:")
    print(
        raw.groupby(["source_task", "target_task"])["false_reuse"]
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()
