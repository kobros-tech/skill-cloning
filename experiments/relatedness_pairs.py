"""Prerequisite-based new-skill acquisition experiment.

Issue #3 asks a different question from the original PR #4 experiment:

```
For a fixed target task, does acquiring relevant prerequisite skills first
make the target easier or more reliable to acquire?
```

Each condition fixes the target and varies the prior-skill history. Prior
skills are genuinely trained before the target is attempted. The target then
uses the corrected PR #2 controller to choose reuse, clone+adapt, or scratch.
A scratch condition has no prior skills and therefore necessarily starts from
fresh initialization.

The primary outcome is acquisition success within a predeclared fixed budget.
Convergence steps and final held-out performance are secondary outcomes.

The experiment intentionally does not assume that the proposed prerequisite
relationships are correct. No cloning advantage is forced: the controller can
choose scratch when prior skills are not sufficiently compatible.

Usage from repository root:
python experiments/relatedness_pairs.py

Outputs:
results/relatedness_pairs.csv
results/relatedness_pairs_summary.csv
results/plot_prerequisite_acquisition.png
"""
from **future** import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(**file**).resolve().parents[1]
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

TASK_SEED_INDEX = {
"addition": 0,
"subtraction": 1,
"multiplication": 2,
"powers": 3,
"squares": 4,
"division": 5,
}

CONDITIONS = [
("subtraction", (), "none"),
("subtraction", ("addition",), "addition"),
("subtraction", ("addition", "multiplication"), "addition+multiplication"),
("division", (), "none"),
("division", ("addition",), "addition"),
("division", ("addition", "multiplication"), "addition+multiplication"),
("squares", (), "none"),
("squares", ("addition",), "addition"),
("squares", ("addition", "multiplication"), "addition+multiplication"),
("powers", (), "none"),
("powers", ("addition",), "addition"),
("powers", ("addition", "multiplication"), "addition+multiplication"),
]

def train_to_accuracy(net, X, y):
"""Train up to the fixed budget and return (steps, success)."""
for epoch in range(1, MAX_EPOCHS + 1):
net.train_step(X, y, lr=LR)
if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
return epoch, True
return MAX_EPOCHS, False

def _task_seed(seed: int, condition_index: int, task: str, offset: int = 0) -> int:
"""Deterministically separate task/history data roles and conditions."""
return seed * 100_000 + condition_index * 1_000 + TASK_SEED_INDEX[task] + offset

def acquire_prior_skills(prior_tasks, seed: int, condition_index: int):
"""Train prerequisites independently; failed prerequisites are not exposed."""
skills = {}
prior_rows = []
history_valid = True

```
for task in prior_tasks:
    task_seed = _task_seed(seed, condition_index, task, offset=10_000)
    X, y = sample_task(task, N_TRAIN, seed=task_seed)
    net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=task_seed)
    net.reset_optimizer()

    steps, success = train_to_accuracy(net, X, y)

    prior_rows.append({
        "prior_task": task,
        "prior_training_steps": steps,
        "prior_acquisition_success": success,
    })

    if success:
        skills[task] = Skill(task, net, origin="scratch", parent=None)
    else:
        # A failed prerequisite is an unavailable history, not an acquired skill.
        history_valid = False

return skills, prior_rows, history_valid
```

def run_condition(
target: str,
prior_tasks: tuple[str, ...],
history_label: str,
seed: int,
condition_index: int,
):
"""Run one target/history/seed condition through the acquisition controller."""
skills, prior_rows, history_valid = acquire_prior_skills(
prior_tasks, seed, condition_index
)

```
row = {
    "target_task": target,
    "prior_history": history_label,
    "prior_tasks": "+".join(prior_tasks) if prior_tasks else "none",
    "seed": seed,
    "prior_history_valid": bool(history_valid),
}

for item in prior_rows:
    prefix = item["prior_task"]
    row[f"{prefix}_prior_steps"] = item["prior_training_steps"]
    row[f"{prefix}_prior_success"] = item["prior_acquisition_success"]

if not history_valid:
    # Fail closed: do not silently downgrade the requested history to a
    # partial history and do not expose an unsuccessful prerequisite to
    # the controller.
    row.update({
        "strategy": "prerequisite_failed",
        "source_task": None,
        "compatibility_score": np.nan,
        "solve_accuracy": np.nan,
        "adaptation_steps": np.nan,
        "fixed_budget": MAX_EPOCHS,
        "target_attempted": False,
        "acquisition_success": False,
        "heldout_accuracy": np.nan,
        "heldout_mse": np.nan,
    })
    return row

controller_seed = _task_seed(
    seed, condition_index, target, offset=20_000
)
decision = comp.decide(
    skills,
    target,
    base_seed=controller_seed,
)

target_train_seed = _task_seed(
    seed, condition_index, target, offset=30_000
)
X_target, y_target = sample_task(
    target,
    N_TRAIN,
    seed=target_train_seed,
)

if decision["action"] == "reuse":
    steps = 0
    success = True
    selected_parent = decision["parent"]
    net = skills[selected_parent].net

elif decision["action"] == "clone":
    selected_parent = decision["parent"]
    net = skills[selected_parent].net.clone()
    net.reset_optimizer()
    steps, success = train_to_accuracy(
        net,
        X_target,
        y_target,
    )

else:
    selected_parent = None
    net = TinyMLP(
        hidden_dim=HIDDEN_DIM,
        seed=target_train_seed,
    )
    net.reset_optimizer()
    steps, success = train_to_accuracy(
        net,
        X_target,
        y_target,
    )

test_seed = _task_seed(
    seed, condition_index, target, offset=40_000
)
X_test, y_test = sample_task(
    target,
    N_EVAL,
    seed=test_seed,
)

test_acc = float(net.accuracy(X_test, y_test, tol=ACC_TOL))
test_mse = float(net.mse(X_test, y_test))

row.update({
    "strategy": decision["action"],
    "source_task": selected_parent,
    "compatibility_score": float(decision["score"]),
    "solve_accuracy": float(decision["solve_accuracy"]),
    "adaptation_steps": steps,
    "fixed_budget": MAX_EPOCHS,
    "target_attempted": True,
    "acquisition_success": bool(success),
    "heldout_accuracy": test_acc,
    "heldout_mse": test_mse,
})

return row
```

def run_all_conditions(n_seeds: int = N_SEEDS):
rows = []

```
for condition_index, (
    target,
    prior_tasks,
    history_label,
) in enumerate(CONDITIONS):
    for seed in range(n_seeds):
        rows.append(
            run_condition(
                target,
                prior_tasks,
                history_label,
                seed,
                condition_index,
            )
        )

return rows
```

def summarize(raw: pd.DataFrame) -> pd.DataFrame:
rows = []

```
for (target, history), group in raw.groupby(
    ["target_task", "prior_history"],
    sort=False,
):
    attempted = group[group["target_attempted"]]

    rows.append({
        "target_task": target,
        "prior_history": history,
        "n_seeds": len(group),
        "prior_history_valid": int(group["prior_history_valid"].sum()),
        "prerequisite_failures": int(
            (~group["prior_history_valid"]).sum()
        ),
        "target_attempts": int(group["target_attempted"].sum()),
        "acquisition_successes": int(
            group["acquisition_success"].sum()
        ),
        "acquisition_success_rate": (
            attempted["acquisition_success"].mean()
            if len(attempted)
            else np.nan
        ),
        "mean_adaptation_steps": (
            attempted["adaptation_steps"].mean()
            if len(attempted)
            else np.nan
        ),
        "std_adaptation_steps": (
            attempted["adaptation_steps"].std()
            if len(attempted)
            else np.nan
        ),
        "mean_heldout_accuracy": (
            attempted["heldout_accuracy"].mean()
            if len(attempted)
            else np.nan
        ),
        "std_heldout_accuracy": (
            attempted["heldout_accuracy"].std()
            if len(attempted)
            else np.nan
        ),
        "mean_heldout_mse": (
            attempted["heldout_mse"].mean()
            if len(attempted)
            else np.nan
        ),
        "std_heldout_mse": (
            attempted["heldout_mse"].std()
            if len(attempted)
            else np.nan
        ),
        "reuse_count": int(
            (group["strategy"] == "reuse").sum()
        ),
        "clone_count": int(
            (group["strategy"] == "clone").sum()
        ),
        "scratch_count": int(
            (group["strategy"] == "scratch").sum()
        ),
        "prerequisite_failed_count": int(
            (group["strategy"] == "prerequisite_failed").sum()
        ),
    })

order = {
    (target, history): i
    for i, (target, _, history) in enumerate(CONDITIONS)
}

summary = pd.DataFrame(rows)
summary["_order"] = [
    order[(r.target_task, r.prior_history)]
    for r in summary.itertuples()
]

return (
    summary
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)
```

def make_plot(summary: pd.DataFrame, out_path: Path):
import matplotlib

```
matplotlib.use("Agg")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

labels = [
    f"{r.target_task}\n{r.prior_history}"
    for r in summary.itertuples()
]

ax.bar(
    range(len(summary)),
    summary["acquisition_success_rate"].fillna(0.0),
)
ax.set_xticks(range(len(summary)))
ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right",
    fontsize=8,
)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Target acquisition success rate (valid histories)")
ax.set_title("New-skill acquisition by prior-skill history")

fig.tight_layout()
fig.savefig(out_path, dpi=140)
plt.close(fig)
```

def main():
raw = pd.DataFrame(run_all_conditions())
summary = summarize(raw)

```
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)

raw.to_csv(
    out_dir / "relatedness_pairs.csv",
    index=False,
)
summary.to_csv(
    out_dir / "relatedness_pairs_summary.csv",
    index=False,
)
make_plot(
    summary,
    out_dir / "plot_prerequisite_acquisition.png",
)

print("Prerequisite-based new-skill acquisition experiment")
print(
    f"Seeds per condition: {N_SEEDS}; "
    f"conditions: {len(CONDITIONS)}"
)
print()
print(summary.to_string(index=False))
```

if **name** == "**main**":
main()
