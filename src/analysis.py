"""
analysis.py — Phase 4 statistics.

All comparisons are paired by seed (same seed -> same random data draws across
strategies), so paired tests (paired t-test + Wilcoxon signed-rank as a
distribution-free check) are used rather than unpaired ones.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats


def paired_compare(df: pd.DataFrame, value_col: str, group_col: str, a: str, b: str,
                    filter_task: str | None = None, task_col: str = "task"):
    """Paired comparison of `value_col` between group a and group b (matched by seed)."""
    d = df if filter_task is None else df[df[task_col] == filter_task]
    pivot = d.pivot(index="seed", columns=group_col, values=value_col)
    xa, xb = pivot[a].values, pivot[b].values
    diff = xa - xb
    t_stat, t_p = stats.ttest_rel(xa, xb)
    try:
        w_stat, w_p = stats.wilcoxon(xa, xb)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {
        "a": a, "b": b, "task": filter_task,
        "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
        "mean_diff (a-b)": float(np.mean(diff)), "std_diff": float(np.std(diff, ddof=1)),
        "n": len(diff),
        "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
    }


def build_report_tables(final: pd.DataFrame, conv: pd.DataFrame, params: pd.DataFrame):
    tables = {}

    # --- Retention / forgetting: final MSE on each task, shared vs clone_and_adapt ---
    rows = []
    for task in final["task"].unique():
        rows.append(paired_compare(final, "final_mse", "strategy",
                                    "shared_sequential", "clone_and_adapt", filter_task=task))
    tables["forgetting_shared_vs_proposed"] = pd.DataFrame(rows)

    # --- Retention: clone_and_adapt vs independent_scratch (should be ~equal: sanity check) ---
    rows = []
    for task in final["task"].unique():
        rows.append(paired_compare(final, "final_mse", "strategy",
                                    "clone_and_adapt", "independent_scratch", filter_task=task))
    tables["retention_proposed_vs_scratch"] = pd.DataFrame(rows)

    # --- Convergence speed: clone_and_adapt vs independent_scratch, per task ---
    rows = []
    for task in conv["task"].unique():
        rows.append(paired_compare(conv, "convergence_steps", "strategy",
                                    "independent_scratch", "clone_and_adapt", filter_task=task))
    tables["convergence_scratch_vs_proposed"] = pd.DataFrame(rows)

    # --- Summary stats tables ---
    tables["final_mse_summary"] = (final.groupby(["strategy", "task"])["final_mse"]
                                    .agg(["mean", "std", "count"]).reset_index())
    tables["final_acc_summary"] = (final.groupby(["strategy", "task"])["final_acc"]
                                    .agg(["mean", "std", "count"]).reset_index())
    tables["convergence_summary"] = (conv.groupby(["strategy", "task"])["convergence_steps"]
                                      .agg(["mean", "std", "count"]).reset_index())
    tables["params_summary"] = (params.groupby("strategy")["total_params"]
                                 .agg(["mean", "std"]).reset_index())
    return tables


if __name__ == "__main__":
    final = pd.read_csv("final.csv")
    conv = pd.read_csv("convergence.csv")
    params = pd.read_csv("params.csv")
    tables = build_report_tables(final, conv, params)
    for name, t in tables.items():
        print("=" * 10, name, "=" * 10)
        print(t.to_string(index=False))
        print()
