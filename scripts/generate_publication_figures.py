"""Generate publication figures directly from workflow CSV outputs.

The figures in this script are deliberately downstream of the authoritative CSVs:
- relatedness_pairs_summary.csv -> fixed-target history figure
- signed_domain_pairs_comparison.csv -> signed-domain paired-speedup figure

No manuscript numbers are hard-coded here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results")


def read_csv(name: str):
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fixed_target_figure() -> None:
    rows = read_csv("relatedness_pairs_summary.csv")
    histories = ["none", "addition", "addition+multiplication"]
    targets = ["subtraction", "division", "squares", "powers"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for target in targets:
        vals = []
        for history in histories:
            row = next(r for r in rows if r["target_task"] == target and r["prior_history"] == history)
            if target == "squares":
                vals.append(float(row["acquisition_success_rate"]) * 100.0)
            else:
                vals.append(float(row["mean_adaptation_steps"]))
        ax.plot(range(len(histories)), vals, marker="o", label=target)

    ax.set_xticks(range(len(histories)), ["None", "Addition", "Addition + multiplication"])
    ax.set_ylabel("Mean budgeted steps (squares: success %)" )
    ax.set_xlabel("Prior-skill history")
    ax.set_title("Fixed-target acquisition under increasing skill history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_fixed_target_history.png", dpi=300)
    fig.savefig(RESULTS / "figure_fixed_target_history.svg")
    plt.close(fig)


def signed_domain_figure() -> None:
    rows = read_csv("signed_domain_pairs_comparison.csv")
    pairs = [r["pair"] for r in rows]
    nonnegative = [float(r["mean_speedup_nonnegative"]) for r in rows]
    signed = [float(r["mean_speedup_signed"]) for r in rows]

    x = list(range(len(pairs)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar([i - width / 2 for i in x], nonnegative, width, label="Non-negative")
    ax.bar([i + width / 2 for i in x], signed, width, label="Signed")
    ax.axhline(1.0, linewidth=1.0, linestyle="--")
    ax.set_xticks(x, [p.replace("->", " → ") for p in pairs], rotation=20, ha="right")
    ax.set_ylabel("Mean speedup (scratch / clone)")
    ax.set_title("Transfer speedup under domain expansion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_signed_domain_speedup.png", dpi=300)
    fig.savefig(RESULTS / "figure_signed_domain_speedup.svg")
    plt.close(fig)


if __name__ == "__main__":
    fixed_target_figure()
    signed_domain_figure()
    print("Generated authoritative publication figures from workflow CSVs.")
