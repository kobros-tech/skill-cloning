import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tasks import TASK_ORDER

plt.rcParams.update({"figure.dpi": 140, "font.size": 10})
COLORS = {"shared_sequential": "#d62728", "independent_scratch": "#7f7f7f", "clone_and_adapt": "#1f77b4"}
LABELS = {"shared_sequential": "Shared (interference)", "independent_scratch": "Independent (scratch)",
          "clone_and_adapt": "Proposed (clone-and-adapt)"}

final = pd.read_csv("final.csv")
conv = pd.read_csv("convergence.csv")
params = pd.read_csv("params.csv")
with open("logs_illustrative.pkl", "rb") as f:
    logs = pickle.load(f)
with open("decisions_illustrative.pkl", "rb") as f:
    decisions = pickle.load(f)

strategies = ["shared_sequential", "independent_scratch", "clone_and_adapt"]

# ---------------------------------------------------------------- Plot 1: forgetting
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(TASK_ORDER))
width = 0.25
for i, s in enumerate(strategies):
    means = [final[(final.strategy == s) & (final.task == t)]["final_mse"].mean() for t in TASK_ORDER]
    stds = [final[(final.strategy == s) & (final.task == t)]["final_mse"].std() for t in TASK_ORDER]
    ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=LABELS[s], color=COLORS[s])
ax.set_yscale("symlog", linthresh=1)
ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in TASK_ORDER])
ax.set_ylabel("Final held-out MSE (log scale)\n← lower = better retained")
ax.set_title("Retention after learning the full curriculum\n(mean ± std over 15 seeds)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plot_forgetting.png")
plt.close(fig)

# ---------------------------------------------------------------- Plot 2: forgetting curve over curriculum steps (illustrative seed)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, s in zip(axes, ["shared_sequential", "clone_and_adapt"]):
    log = logs[s]
    for t in TASK_ORDER:
        sub = log[log.evaluated_on == t].sort_values("curriculum_step")
        ax.plot(sub["curriculum_step"], sub["mse"] + 1e-3, marker="o", label=t.capitalize())
    ax.set_yscale("log")
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([t[:4] for t in TASK_ORDER], rotation=0)
    ax.set_xlabel("Curriculum step (task just trained)")
    ax.set_title(LABELS[s])
axes[0].set_ylabel("MSE on task (log scale)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("How each previously-learned task's error evolves as new tasks arrive (seed 0)")
fig.tight_layout()
fig.savefig("plot_forgetting_curve.png")
plt.close(fig)

# ---------------------------------------------------------------- Plot 3: convergence speed, clone vs scratch
fig, ax = plt.subplots(figsize=(7, 4.2))
comp_tasks = TASK_ORDER  # show all 4
x = np.arange(len(comp_tasks))
width = 0.35
for i, s in enumerate(["independent_scratch", "clone_and_adapt"]):
    means = [conv[(conv.strategy == s) & (conv.task == t)]["convergence_steps"].mean() for t in comp_tasks]
    stds = [conv[(conv.strategy == s) & (conv.task == t)]["convergence_steps"].std() for t in comp_tasks]
    ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3, label=LABELS[s], color=COLORS[s])
ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in comp_tasks])
ax.set_ylabel("Epochs to reach 85% train-batch accuracy")
ax.set_title("Convergence speed: cloned initialization vs. random initialization\n(mean ± std over 15 seeds)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plot_convergence.png")
plt.close(fig)

# ---------------------------------------------------------------- Plot 4: compatibility scores / decisions (illustrative)
fig, ax = plt.subplots(figsize=(7, 4))
tasks_ = list(decisions.keys())
scores = [decisions[t]["score"] for t in tasks_]
actions = [decisions[t]["action"] for t in tasks_]
action_color = {"scratch": "#7f7f7f", "clone": "#1f77b4", "reuse": "#2ca02c"}
bars = ax.bar(tasks_, scores, color=[action_color[a] for a in actions])
ax.axhline(0.90, color="green", linestyle="--", linewidth=1, label=r"$\tau_{solve}=0.90$ (reuse)")
ax.axhline(0.15, color="blue", linestyle="--", linewidth=1, label=r"$\tau_{clone}=0.15$ (clone)")
for bar, a, t in zip(bars, actions, tasks_):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, a, ha="center", fontsize=8)
    if decisions[t]["parent"]:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f"<- {decisions[t]['parent']}",
                 ha="center", fontsize=7, rotation=90, color="white")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Compatibility score of best-matching existing skill")
ax.set_title("Skill-selection decision for each incoming task (seed 0)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plot_decisions.png")
plt.close(fig)

# ---------------------------------------------------------------- Plot 5: parameter growth
fig, ax = plt.subplots(figsize=(5, 4))
means = [params[params.strategy == s]["total_params"].mean() for s in strategies]
ax.bar([LABELS[s] for s in strategies], means, color=[COLORS[s] for s in strategies])
ax.set_ylabel("Total parameters across all skills")
ax.set_title("Parameter growth after 4 tasks")
plt.xticks(rotation=15, ha="right")
fig.tight_layout()
fig.savefig("plot_params.png")
plt.close(fig)

print("done")
