"""
experiment.py — Phase 4 (Evaluation) driver.

Runs run_shared / run_scratch / run_proposed across N_SEEDS random seeds and
assembles tidy pandas DataFrames covering:
  - final MSE / accuracy per task per strategy per seed  (retention + performance)
  - convergence steps per task per strategy per seed      (training speed)
  - total parameter count per strategy per seed           (parameter growth)
  - the full per-step forgetting log for one illustrative seed (for a plot)
  - the decision trace (reuse/clone/scratch + compatibility score) for one seed
"""
from __future__ import annotations
import pandas as pd
from tasks import TASK_ORDER
import strategies as strat

N_SEEDS = 15
ILLUSTRATIVE_SEED = 0


def run_all(n_seeds: int = N_SEEDS):
    final_rows = []
    conv_rows = []
    param_rows = []
    logs_illustrative = {}
    decisions_illustrative = None

    for seed in range(n_seeds):
        for run_fn, label in [(strat.run_shared, "shared_sequential"),
                               (strat.run_scratch, "independent_scratch"),
                               (strat.run_proposed, "clone_and_adapt")]:
            result = run_fn(seed)
            for t in TASK_ORDER:
                final_rows.append({
                    "seed": seed, "strategy": label, "task": t,
                    "final_mse": result["final_mse"][t],
                    "final_acc": result["final_acc"][t],
                })
                conv_rows.append({
                    "seed": seed, "strategy": label, "task": t,
                    "convergence_steps": result["convergence_steps"][t],
                })
            param_rows.append({"seed": seed, "strategy": label, "total_params": result["total_params"]})

            if seed == ILLUSTRATIVE_SEED:
                logs_illustrative[label] = pd.DataFrame(result["log"])
                if label == "clone_and_adapt":
                    decisions_illustrative = result["decisions"]

    final_df = pd.DataFrame(final_rows)
    conv_df = pd.DataFrame(conv_rows)
    param_df = pd.DataFrame(param_rows)
    return {
        "final": final_df,
        "convergence": conv_df,
        "params": param_df,
        "logs_illustrative": logs_illustrative,
        "decisions_illustrative": decisions_illustrative,
    }


if __name__ == "__main__":
    out = run_all()
    print(out["final"].groupby(["strategy", "task"])["final_mse"].mean())
