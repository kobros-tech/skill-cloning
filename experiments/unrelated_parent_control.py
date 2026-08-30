"""Matched control for distinguishing relevant transfer from generic pretraining.

For each target, compare scratch learning with cloning from the controller's
selected parent and cloning from a deliberately unrelated previously learned
skill. This is an exploratory control; it uses the same target data, optimizer,
architecture, budget, and seed for all arms.

Results are written to results/unrelated_parent_control.csv when run directly.
"""
from __future__ import annotations

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from skill import TinyMLP
from tasks import TASK_ORDER, sample_task
import strategies as strat
import compatibility as comp

N_SEEDS = 15


def train(net, X, y):
    _, steps = strat._train_track_accuracy(
        net, X, y, strat.EPOCHS, strat.LR, strat.ACC_TARGET
    )
    return steps


def run(n_seeds=N_SEEDS):
    rows = []
    for seed in range(n_seeds):
        # Build the same acquired history used by the proposed controller.
        skills = {}
        for step, task in enumerate(TASK_ORDER):
            X, y = sample_task(task, strat.N_TRAIN, seed=seed * 100 + step)
            net = TinyMLP(hidden_dim=strat.HIDDEN_DIM, seed=seed * 100 + step)
            train(net, X, y)
            skills[task] = net

        # Only targets with a distinct earlier task are included.
        for target in TASK_ORDER[1:]:
            X, y = sample_task(target, strat.N_TRAIN, seed=seed * 100 + TASK_ORDER.index(target))
            decision = comp.decide({
                name: type("S", (), {"name": name, "net": net})()
                for name, net in skills.items()
                if name != target
            }, target, base_seed=seed * 100 + TASK_ORDER.index(target))

            relevant = decision["parent"]
            candidates = [t for t in TASK_ORDER if t != target and t != relevant and t in skills]
            if not candidates:
                continue
            unrelated = candidates[0]

            scratch = TinyMLP(hidden_dim=strat.HIDDEN_DIM, seed=seed * 100 + TASK_ORDER.index(target))
            clone_relevant = skills[relevant].clone() if hasattr(skills[relevant], "clone") else skills[relevant].clone()
            clone_unrelated = skills[unrelated].clone() if hasattr(skills[unrelated], "clone") else skills[unrelated].clone()
            for net, arm in [(scratch, "scratch"), (clone_relevant, "relevant_clone"), (clone_unrelated, "unrelated_clone")]:
                net.reset_optimizer()
                steps = train(net, X, y)
                rows.append({
                    "seed": seed,
                    "target": target,
                    "relevant_parent": relevant,
                    "unrelated_parent": unrelated,
                    "arm": arm,
                    "convergence_steps": steps,
                    "final_mse": net.mse(X, y),
                    "final_acc": net.accuracy(X, y, tol=strat.ACC_TOL),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = run()
    path = os.path.join(ROOT, "results", "unrelated_parent_control.csv")
    out.to_csv(path, index=False)
    print(out.groupby(["target", "arm"])["convergence_steps"].agg(["mean", "count"]))
    print(f"Saved {path}")
