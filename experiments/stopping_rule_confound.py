"""Controlled experiment for the stopping-rule confound.

The Phase 4 result on powers showed faster convergence for clone-and-adapt but
worse held-out MSE than scratch. The original convergence metric stops training
when training accuracy reaches 85%, so a faster clone can receive fewer updates
and still satisfy the stopping criterion.

This experiment separates the stopping rule from clone quality by evaluating
clone and scratch at the SAME fixed training budgets. It also records the
original 85%-accuracy stopping outcome for reference.

Usage from the repository root:
    python experiments/stopping_rule_confound.py

Outputs:
    results/stopping_rule_confound.csv
    results/stopping_rule_confound_summary.csv

No test-set result is used to choose a budget. Budgets are fixed in advance.
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

from skill import TinyMLP  # noqa: E402
from tasks import sample_task  # noqa: E402

N_SEEDS = 15
N_TRAIN = 200
N_EVAL = 300
HIDDEN_DIM = 32
LR = 0.02
PARENT_EPOCHS = 1500
ACC_TOL = 0.5
ACC_TARGET = 0.85

# Fixed before inspecting results. Includes the range around the original
# convergence points while remaining small enough for a quick CPU experiment.
FIXED_BUDGETS = (50, 100, 200, 400, 800, 1200, 1500)


def train_to_accuracy(net, X, y, max_epochs=PARENT_EPOCHS):
    """Train with the original 85%-accuracy rule and return epochs used."""
    for epoch in range(1, max_epochs + 1):
        net.train_step(X, y, lr=LR)
        if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
            return epoch
    return max_epochs


def train_fixed_budget(net, X, y, budgets):
    """Return train/validation/test metrics after each predeclared budget."""
    X_val, y_val = sample_task("powers", N_EVAL, seed=900_000)
    X_test, y_test = sample_task("powers", N_EVAL, seed=910_000)
    rows = []
    budget_set = set(budgets)
    for epoch in range(1, max(budgets) + 1):
        net.train_step(X, y, lr=LR)
        if epoch in budget_set:
            rows.append(
                {
                    "epoch_budget": epoch,
                    "train_mse": net.mse(X, y),
                    "validation_mse": net.mse(X_val, y_val),
                    "test_mse": net.mse(X_test, y_test),
                    "train_accuracy": net.accuracy(X, y, tol=ACC_TOL),
                }
            )
    return rows


def main():
    rows = []
    stopping_rows = []

    for seed in range(N_SEEDS):
        # First learn multiplication exactly as the Phase 4 curriculum does.
        X_mul, y_mul = sample_task("multiplication", N_TRAIN, seed=seed * 100 + 2)
        parent = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + 2)
        parent.reset_optimizer()
        train_to_accuracy(parent, X_mul, y_mul)

        # The target task is powers; clone starts from the learned multiplication
        # parameters while scratch starts from an independent random init.
        X_pow, y_pow = sample_task("powers", N_TRAIN, seed=seed * 100 + 3)
        clone = parent.clone()
        clone.reset_optimizer()
        scratch = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + 3)
        scratch.reset_optimizer()

        # Reference the original stopping rule without using its result to select
        # the fixed budgets.
        clone_stop = train_to_accuracy(clone, X_pow, y_pow)
        scratch_stop = train_to_accuracy(scratch, X_pow, y_pow)
        stopping_rows.extend(
            [
                {
                    "seed": seed,
                    "strategy": "clone",
                    "stopping_rule": "train_accuracy>=0.85",
                    "epochs_used": clone_stop,
                },
                {
                    "seed": seed,
                    "strategy": "scratch",
                    "stopping_rule": "train_accuracy>=0.85",
                    "epochs_used": scratch_stop,
                },
            ]
        )

        # Re-create both models so fixed-budget measurements start from exactly
        # the same initial states and are not affected by the reference run.
        clone = parent.clone()
        clone.reset_optimizer()
        scratch = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + 3)
        scratch.reset_optimizer()

        for strategy, net in (("clone", clone), ("scratch", scratch)):
            for metrics in train_fixed_budget(net, X_pow, y_pow, FIXED_BUDGETS):
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "source_task": "multiplication" if strategy == "clone" else "random",
                        "target_task": "powers",
                        **metrics,
                    }
                )

    raw = pd.DataFrame(rows)
    stopping = pd.DataFrame(stopping_rows)

    # Paired-by-seed summary at every identical budget.
    summary = (
        raw.groupby(["epoch_budget", "strategy"])[
            ["train_mse", "validation_mse", "test_mse", "train_accuracy"]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw.to_csv(out_dir / "stopping_rule_confound.csv", index=False)
    summary.to_csv(out_dir / "stopping_rule_confound_summary.csv", index=False)
    stopping.to_csv(out_dir / "stopping_rule_confound_stopping_epochs.csv", index=False)

    print("Fixed-budget stopping-rule confound experiment")
    print(f"Seeds: {N_SEEDS}; budgets: {list(FIXED_BUDGETS)}")
    print("\nOriginal 85%-accuracy stopping epochs:")
    print(stopping.groupby("strategy")["epochs_used"].agg(["mean", "std", "min", "max"]))
    print("\nFixed-budget test MSE:")
    print(raw.groupby(["epoch_budget", "strategy"])["test_mse"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
