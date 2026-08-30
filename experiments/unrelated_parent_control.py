"""Matched control separating relevant transfer from generic pretraining.

Only skills acquired before a target are eligible as parents. The controller's
highest-compatibility earlier skill is compared with an unrelated earlier
skill and scratch initialization. If the controller itself selects scratch,
there is still a well-defined highest-compatibility candidate for the control;
this avoids treating ``None`` as a skill while preserving the distinction
between compatibility ranking and the final controller action.

The experiment reports both descriptive arm summaries and paired seed-level
comparisons. The latter are the appropriate inferential unit because all three
arms use the same target, seed, data, and training budget within a comparison.
"""
from __future__ import annotations

import os
import sys
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from skill import TinyMLP, Skill
from tasks import TASK_ORDER, sample_task
import strategies as strat
import compatibility as comp

N_SEEDS = 15


def train(net, X, y):
    _, steps = strat._train_track_accuracy(net, X, y, strat.EPOCHS, strat.LR, strat.ACC_TARGET)
    return steps


def paired_summary(out: pd.DataFrame) -> pd.DataFrame:
    """Compute matched-seed statistics for relevant/unrelated/scratch arms.

    The test unit is a seed within a target. We report the mean paired
    difference, standard deviation, paired t-test, and Wilcoxon signed-rank
    test. No significance claim is hard-coded into the experiment.
    """
    rows = []
    for target, group in out.groupby("target"):
        wide = group.pivot(index="seed", columns="arm", values="convergence_steps").dropna()
        for left, right, label in [
            ("relevant_clone", "scratch", "relevant_minus_scratch"),
            ("relevant_clone", "unrelated_clone", "relevant_minus_unrelated"),
            ("unrelated_clone", "scratch", "unrelated_minus_scratch"),
        ]:
            if wide.empty:
                continue
            diff = wide[left] - wide[right]
            t_stat, t_p = stats.ttest_rel(wide[left], wide[right])
            if (diff != 0).any():
                w_stat, w_p = stats.wilcoxon(wide[left], wide[right], alternative="two-sided")
            else:
                w_stat, w_p = 0.0, 1.0
            rows.append({
                "target": target,
                "comparison": label,
                "n_matched": int(len(diff)),
                "mean_difference_steps": float(diff.mean()),
                "std_difference_steps": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "mean_absolute_difference_steps": float(diff.abs().mean()),
                "paired_t_stat": float(t_stat),
                "paired_t_p": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
            })
    return pd.DataFrame(rows)


def run(n_seeds=N_SEEDS):
    rows = []
    for seed in range(n_seeds):
        skills: dict[str, Skill] = {}
        for step, target in enumerate(TASK_ORDER):
            # Never expose the target or future tasks to parent selection.
            if len(skills) >= 2:
                X, y = sample_task(target, strat.N_TRAIN, seed=seed * 100 + step)
                decision = comp.decide(skills, target, base_seed=seed * 100 + step)
                ranking = decision["ranking"]
                relevant = ranking[0][0]
                unrelated = next(name for name in skills if name != relevant)

                arms = {
                    "scratch": TinyMLP(hidden_dim=strat.HIDDEN_DIM, seed=seed * 100 + step),
                    "relevant_clone": skills[relevant].net.clone(),
                    "unrelated_clone": skills[unrelated].net.clone(),
                }
                for arm, net in arms.items():
                    net.reset_optimizer()
                    steps = train(net, X, y)
                    rows.append({
                        "seed": seed,
                        "target": target,
                        "controller_action": decision["action"],
                        "controller_parent": decision["parent"],
                        "relevant_parent": relevant,
                        "unrelated_parent": unrelated,
                        "arm": arm,
                        "convergence_steps": steps,
                        "final_mse": net.mse(X, y),
                        "final_acc": net.accuracy(X, y, tol=strat.ACC_TOL),
                    })

            # Independently acquire this task to construct the history for the
            # next curriculum step; this is not one of the comparison arms.
            X_hist, y_hist = sample_task(target, strat.N_TRAIN, seed=seed * 100 + step)
            hist_net = TinyMLP(hidden_dim=strat.HIDDEN_DIM, seed=seed * 100 + step)
            train(hist_net, X_hist, y_hist)
            skills[target] = Skill(target, hist_net, origin="scratch")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = run()
    path = os.path.join(ROOT, "results", "unrelated_parent_control.csv")
    out.to_csv(path, index=False)
    summary = paired_summary(out)
    summary_path = os.path.join(ROOT, "results", "unrelated_parent_control_stats.csv")
    summary.to_csv(summary_path, index=False)
    print(out.groupby(["target", "arm"])["convergence_steps"].agg(["mean", "std", "count"]))
    print("\nPaired comparisons:")
    print(summary.to_string(index=False))
    print(f"Saved {path}")
    print(f"Saved {summary_path}")
