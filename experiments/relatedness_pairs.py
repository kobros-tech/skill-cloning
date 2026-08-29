"""Broaden the relatedness evidence across several source -> target task pairs.

PR #3 scientific question (issue #3):
    Does the transfer benefit depend systematically on the relationship
    between source and target tasks, rather than being an isolated property
    of one task pair?

This experiment reuses exactly the training setup, convergence criterion
(85% training-batch accuracy, tol=0.5), optimizer, and seed protocol already
established by experiments/squares_relatedness.py and
experiments/stopping_rule_confound.py, and extends it to a small, deliberately
chosen set of source -> target pairs spanning a range of relatedness levels:

    multiplication -> squares          (strong: squares is a special case of
                                         multiplication, a*a)
    multiplication -> powers           (moderate: powers generalizes
                                         multiplication to repeated products)
    addition       -> subtraction      (weak: same input structure, opposite
                                         operation -- flagged in PR #1/#2 as a
                                         potential negative-transfer case)
    addition       -> multiplication   (null/unrelated control: different
                                         operation and different structure)

For every pair we also record the frozen-parent compatibility score
(compatibility.compatibility_score) as a relatedness proxy, so the analysis
can ask whether larger relatedness scores associate with larger paired
speedups -- without claiming this establishes a universal law (see README).

This experiment measures convergence speed only. It does not modify the main
Phase 4 curriculum, does not touch the compatibility-gated controller
(deliberately -- isolating "does a related parent speed up training" from
"does the controller correctly decide when to reuse/clone" is the point), and
never uses held-out test data to pick a stopping budget.

Usage from the repository root:
    python experiments/relatedness_pairs.py

Output:
    results/relatedness_pairs.csv
    results/relatedness_pairs_summary.csv
    results/plot_relatedness_vs_speedup.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compatibility import compatibility_score  # noqa: E402
from skill import TinyMLP  # noqa: E402
from tasks import sample_task  # noqa: E402

N_SEEDS = 15
N_TRAIN = 200
HIDDEN_DIM = 32
LR = 0.02
MAX_EPOCHS = 1500
ACC_TOL = 0.5
ACC_TARGET = 0.85

# Same seed-index convention already used by squares_relatedness.py and
# compatibility_calibration.py (addition=0, subtraction=1, multiplication=2,
# powers=3; squares reuses index 3, matching PR #2 -- it is never paired
# against powers within the same seed call, so there is no collision).
TASK_SEED_INDEX = {
    "addition": 0,
    "subtraction": 1,
    "multiplication": 2,
    "powers": 3,
    "squares": 3,
}

# A compact set of qualitatively different relatedness levels, per issue #3:
# prefer a small number of deliberately chosen pairs over many arbitrary ones.
PAIRS = [
    ("multiplication", "squares", "strong"),
    ("multiplication", "powers", "moderate"),
    ("addition", "subtraction", "weak"),
    ("addition", "multiplication", "null/unrelated"),
]


def train_to_accuracy(net, X, y):
    """Return the first epoch at which the 85% training-accuracy rule is reached."""
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train_step(X, y, lr=LR)
        if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
            return epoch
    return MAX_EPOCHS


def run_pair(source: str, target: str, relatedness_label: str, n_seeds: int = N_SEEDS):
    """Run the clone-vs-scratch convergence comparison for one source->target pair.

    Returns a list of per-seed row dicts. Kept separate from main() so it can
    be exercised directly (with a small n_seeds) by the regression test.
    """
    src_idx = TASK_SEED_INDEX[source]
    tgt_idx = TASK_SEED_INDEX[target]
    rows = []
    for seed in range(n_seeds):
        X_src, y_src = sample_task(source, N_TRAIN, seed=seed * 100 + src_idx)
        parent = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + src_idx)
        parent.reset_optimizer()
        parent_epochs = train_to_accuracy(parent, X_src, y_src)

        # Frozen-parent relatedness proxy, evaluated before any target-task
        # training touches the clone (same probe used by the controller).
        rel_score = compatibility_score(parent, target, seed * 100 + tgt_idx)

        X_tgt, y_tgt = sample_task(target, N_TRAIN, seed=seed * 100 + tgt_idx)

        clone = parent.clone()
        clone.reset_optimizer()
        scratch = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + tgt_idx)
        scratch.reset_optimizer()

        clone_epochs = train_to_accuracy(clone, X_tgt, y_tgt)
        scratch_epochs = train_to_accuracy(scratch, X_tgt, y_tgt)

        rows.append({
            "pair": f"{source}->{target}",
            "relatedness_label": relatedness_label,
            "source_task": source,
            "target_task": target,
            "seed": seed,
            "relatedness_score": rel_score,
            "parent_epochs": parent_epochs,
            "clone_epochs": clone_epochs,
            "scratch_epochs": scratch_epochs,
            "speedup": scratch_epochs / clone_epochs,
        })
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for pair, group in raw.groupby("pair", sort=False):
        relatedness_label = group["relatedness_label"].iloc[0]
        t_stat, t_p = stats.ttest_rel(group["scratch_epochs"], group["clone_epochs"])
        summary_rows.append({
            "pair": pair,
            "relatedness_label": relatedness_label,
            "n_seeds": len(group),
            "mean_relatedness_score": group["relatedness_score"].mean(),
            "mean_clone_epochs": group["clone_epochs"].mean(),
            "std_clone_epochs": group["clone_epochs"].std(),
            "mean_scratch_epochs": group["scratch_epochs"].mean(),
            "std_scratch_epochs": group["scratch_epochs"].std(),
            "mean_speedup": group["speedup"].mean(),
            "std_speedup": group["speedup"].std(),
            "min_speedup": group["speedup"].min(),
            "max_speedup": group["speedup"].max(),
            "paired_t_stat": t_stat,
            "paired_t_p": t_p,
        })
    # Preserve the deliberate strong -> null ordering from PAIRS rather than
    # whatever order groupby happens to produce.
    order = [f"{s}->{t}" for s, t, _ in PAIRS]
    summary = pd.DataFrame(summary_rows)
    summary["_order"] = summary["pair"].map({p: i for i, p in enumerate(order)})
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def make_plot(summary: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = summary["mean_relatedness_score"]
    y = summary["mean_speedup"]
    yerr = summary["std_speedup"]
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, color="#1f77b4", markersize=8)
    for _, row in summary.iterrows():
        ax.annotate(row["pair"], (row["mean_relatedness_score"], row["mean_speedup"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="no speedup (clone == scratch)")
    ax.set_xlabel("Mean frozen-parent relatedness score (compatibility_score)")
    ax.set_ylabel("Mean paired speedup (scratch epochs / clone epochs)")
    ax.set_title("Transfer speedup vs. source-target relatedness\n(mean \u00b1 std over 15 seeds per pair)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    all_rows = []
    for source, target, relatedness_label in PAIRS:
        all_rows.extend(run_pair(source, target, relatedness_label))

    raw = pd.DataFrame(all_rows)
    summary = summarize(raw)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw.to_csv(out_dir / "relatedness_pairs.csv", index=False)
    summary.to_csv(out_dir / "relatedness_pairs_summary.csv", index=False)
    make_plot(summary, out_dir / "plot_relatedness_vs_speedup.png")

    print("Relatedness pairs experiment")
    print(f"Seeds per pair: {N_SEEDS}; pairs: {[p for p, _, _ in PAIRS]}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
