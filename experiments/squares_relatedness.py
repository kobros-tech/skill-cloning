"""Test whether a related intermediate task produces transfer-speedup evidence.

This experiment adds ``squares`` as a deliberately separate diagnostic task.
The source task is multiplication and the target task is squares, so the
comparison asks whether a closely related parent gives clone-and-adapt a
convergence advantage over an independently initialized model.

The experiment intentionally measures convergence speed only. It does not
modify the main Phase 4 curriculum and does not use the test set to choose
hyperparameters or stopping budgets.

Usage from the repository root:
    python experiments/squares_relatedness.py

Output:
    results/squares_relatedness.csv
    results/squares_relatedness_summary.csv
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
HIDDEN_DIM = 32
LR = 0.02
MAX_EPOCHS = 1500
ACC_TOL = 0.5
ACC_TARGET = 0.85


def train_to_accuracy(net, X, y):
    """Return the first epoch at which the original 85% rule is reached."""
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train_step(X, y, lr=LR)
        if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
            return epoch
    return MAX_EPOCHS


def main():
    rows = []

    for seed in range(N_SEEDS):
        # Learn the parent skill using the same training setup as the earlier
        # curriculum, then clone it for the related target task.
        X_mul, y_mul = sample_task("multiplication", N_TRAIN, seed=seed * 100 + 2)
        parent = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + 2)
        parent.reset_optimizer()
        parent_epochs = train_to_accuracy(parent, X_mul, y_mul)

        X_sq, y_sq = sample_task("squares", N_TRAIN, seed=seed * 100 + 3)

        clone = parent.clone()
        clone.reset_optimizer()
        scratch = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + 3)
        scratch.reset_optimizer()

        clone_epochs = train_to_accuracy(clone, X_sq, y_sq)
        scratch_epochs = train_to_accuracy(scratch, X_sq, y_sq)

        rows.extend(
            [
                {
                    "seed": seed,
                    "strategy": "clone",
                    "source_task": "multiplication",
                    "target_task": "squares",
                    "parent_epochs": parent_epochs,
                    "epochs_to_85pct": clone_epochs,
                },
                {
                    "seed": seed,
                    "strategy": "scratch",
                    "source_task": "random",
                    "target_task": "squares",
                    "parent_epochs": np.nan,
                    "epochs_to_85pct": scratch_epochs,
                },
            ]
        )

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("strategy")["epochs_to_85pct"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    # Report the paired per-seed speedup so the effect can be compared directly
    # with the earlier relatedness experiments.
    paired = raw.pivot(index="seed", columns="strategy", values="epochs_to_85pct")
    paired["scratch_over_clone"] = paired["scratch"] / paired["clone"]
    summary.loc[len(summary)] = [
        "paired_speedup",
        paired["scratch_over_clone"].mean(),
        paired["scratch_over_clone"].std(),
        paired["scratch_over_clone"].min(),
        paired["scratch_over_clone"].max(),
    ]

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw.to_csv(out_dir / "squares_relatedness.csv", index=False)
    summary.to_csv(out_dir / "squares_relatedness_summary.csv", index=False)

    print("Squares relatedness experiment")
    print(f"Seeds: {N_SEEDS}; target: squares; source: multiplication")
    print("\nConvergence epochs:")
    print(summary.to_string(index=False))
    print("\nPer-seed scratch/clone convergence ratio:")
    print(paired["scratch_over_clone"].to_string())


if __name__ == "__main__":
    main()
