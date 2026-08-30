"""Signed-domain transfer-robustness / domain-sensitivity analysis.

Research question (primary):
    Does the source-target transfer behavior observed in the existing
    (non-negative-domain) experiments persist when the task distribution is
    expanded to include negative operands?

Secondary question:
    Does expanding the input domain change which prior skills are useful,
    neutral, or harmful?

This is a robustness/domain-sensitivity study, not an attempt to reproduce a
particular preferred result. If signed-domain results differ from the
non-negative baseline, that difference is itself the finding -- it means
transfer depends on the input distribution rather than being a fixed
property of the source-target task pair. See docs/final_paper.md's
"Domain-sensitivity analysis" section for the full interpretation.

Design, matching the plan this experiment implements:
  - Only the operand domain changes between conditions. Architecture, hidden
    size, optimizer, learning rate, initialization, batch size (N_TRAIN),
    stopping criterion, training budget, seed protocol, compatibility
    calculation, controller thresholds, and data-role separation are all
    identical to the existing experiments (relatedness_pairs.py,
    squares_relatedness.py, stopping_rule_confound.py).
  - The non-negative domain is tasks.py's default ("nonnegative") and is
    untouched -- every existing result remains reproducible byte-for-bit
    (verified separately; see the tasks.py commit).
  - Matched seeds: seed r uses the identical seed in both domains, so every
    comparison below is paired by seed, not just by condition.
  - Powers keeps its exponent non-negative in both domains (a negative
    exponent introduces fractional targets, which would confound operand
    sign with a change in the target's numeric structure).
  - Division's divisor is never zero in either domain (see tasks.py).

Two experiments are run, both under both domains with the same seeds:

  A. The four scientifically load-bearing relatedness pairs from the
     existing paper: multiplication->powers, multiplication->squares,
     addition->subtraction, addition->multiplication. Clone vs. scratch,
     forced clone (matching relatedness_pairs.py's ancestor,
     squares_relatedness.py / stopping_rule_confound.py's methodology) --
     this isolates "does a related parent still speed up training" from
     "does the controller still decide to use it", which is answered
     separately by experiment B.

  B. The fixed-target prerequisite-history matrix (subtraction / division /
     squares / powers, under no-prior / addition / addition+multiplication),
     using the actual compatibility-gated controller
     (compatibility.decide()) -- the same design as
     experiments/relatedness_pairs.py, parameterized by domain.

Sign-specific diagnostics (Section 10 of the plan) are computed for the
three pairs where a domain effect is most plausible on structural grounds:
multiplication->powers, multiplication->squares, addition->subtraction.

Usage from the repository root:
    python experiments/signed_domain_transfer.py

Output:
    results/signed_domain_pairs.csv
    results/signed_domain_pairs_summary.csv
    results/signed_domain_history.csv
    results/signed_domain_history_summary.csv
    results/signed_domain_sign_breakdown.csv
    results/plot_signed_domain_speedup.png
    results/plot_signed_domain_success_rate.png
    results/plot_signed_domain_compatibility_vs_speedup.png
    results/plot_signed_domain_convergence.png
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

import compatibility as comp  # noqa: E402
from skill import Skill, TinyMLP  # noqa: E402
from tasks import DOMAINS, sample_task  # noqa: E402

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

PAIRS = [
    ("multiplication", "powers", "moderate/strong"),
    ("multiplication", "squares", "strong (structural)"),
    ("addition", "subtraction", "weak"),
    ("addition", "multiplication", "null/unrelated"),
]

HISTORY_CONDITIONS = [
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

# Pairs diagnosed for sign-specific effects (Section 10 of the plan).
SIGN_DIAGNOSTIC_PAIRS = [
    ("multiplication", "powers"),
    ("multiplication", "squares"),
    ("addition", "subtraction"),
]


def train_to_accuracy(net, X, y, max_epochs: int = MAX_EPOCHS):
    """Train up to max_epochs. Returns (epochs_taken, reached_target: bool)."""
    for epoch in range(1, max_epochs + 1):
        net.train_step(X, y, lr=LR)
        if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
            return epoch, True
    return max_epochs, False


def _task_seed(seed: int, salt: int, task: str, offset: int = 0) -> int:
    return seed * 100_000 + salt * 1_000 + TASK_SEED_INDEX[task] + offset


# --------------------------------------------------------------------------
# Experiment A: forced-clone relatedness pairs, both domains, matched seeds.
# --------------------------------------------------------------------------

def run_pair_domain(source: str, target: str, relatedness_label: str, domain: str, seed: int) -> dict:
    src_seed = _task_seed(seed, 1, source, offset=0)
    tgt_seed = _task_seed(seed, 1, target, offset=0)

    X_src, y_src = sample_task(source, N_TRAIN, seed=src_seed, domain=domain)
    parent = TinyMLP(hidden_dim=HIDDEN_DIM, seed=src_seed)
    parent.reset_optimizer()
    parent_epochs, parent_reached = train_to_accuracy(parent, X_src, y_src)

    rel_score = comp.compatibility_score(parent, target, tgt_seed, domain=domain)
    solve_acc = comp.solve_probe_accuracy(parent, target, tgt_seed, domain=domain)

    X_tgt, y_tgt = sample_task(target, N_TRAIN, seed=tgt_seed, domain=domain)

    clone = parent.clone()
    clone.reset_optimizer()
    scratch = TinyMLP(hidden_dim=HIDDEN_DIM, seed=tgt_seed)
    scratch.reset_optimizer()

    clone_epochs, clone_success = train_to_accuracy(clone, X_tgt, y_tgt)
    scratch_epochs, scratch_success = train_to_accuracy(scratch, X_tgt, y_tgt)

    X_hold, y_hold = sample_task(target, N_EVAL, seed=tgt_seed + 500_000, domain=domain)
    clone_heldout_mse = clone.mse(X_hold, y_hold)
    scratch_heldout_mse = scratch.mse(X_hold, y_hold)

    return {
        "domain": domain,
        "pair": f"{source}->{target}",
        "relatedness_label": relatedness_label,
        "source_task": source,
        "target_task": target,
        "seed": seed,
        "relatedness_score": rel_score,
        "solve_accuracy": solve_acc,
        "parent_epochs": parent_epochs,
        "parent_reached": parent_reached,
        "clone_epochs": clone_epochs,
        "clone_success": clone_success,
        "scratch_epochs": scratch_epochs,
        "scratch_success": scratch_success,
        "speedup": scratch_epochs / clone_epochs,
        "clone_heldout_mse": clone_heldout_mse,
        "scratch_heldout_mse": scratch_heldout_mse,
    }


def run_all_pairs(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for source, target, label in PAIRS:
            for seed in range(n_seeds):
                rows.append(run_pair_domain(source, target, label, domain, seed))
    return pd.DataFrame(rows)


def summarize_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, pair), group in raw.groupby(["domain", "pair"], sort=False):
        rows.append({
            "domain": domain,
            "pair": pair,
            "relatedness_label": group["relatedness_label"].iloc[0],
            "n_seeds": len(group),
            "mean_relatedness_score": group["relatedness_score"].mean(),
            "mean_clone_epochs": group["clone_epochs"].mean(),
            "std_clone_epochs": group["clone_epochs"].std(),
            "mean_scratch_epochs": group["scratch_epochs"].mean(),
            "std_scratch_epochs": group["scratch_epochs"].std(),
            "clone_success_rate": group["clone_success"].mean(),
            "scratch_success_rate": group["scratch_success"].mean(),
            "mean_speedup": group["speedup"].mean(),
            "std_speedup": group["speedup"].std(),
            "median_speedup": group["speedup"].median(),
        })
    order = {p: i for i, (s, t, _) in enumerate(PAIRS) for p in [f"{s}->{t}"]}
    summary = pd.DataFrame(rows)
    summary["_o1"] = summary["domain"].map({d: i for i, d in enumerate(DOMAINS)})
    summary["_o2"] = summary["pair"].map(order)
    return summary.sort_values(["_o2", "_o1"]).drop(columns=["_o1", "_o2"]).reset_index(drop=True)


def paired_domain_comparison_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    """For each pair, compare non-negative vs signed speedup, paired by seed."""
    rows = []
    for pair in raw["pair"].unique():
        sub = raw[raw["pair"] == pair]
        nn = sub[sub["domain"] == "nonnegative"].set_index("seed")
        sg = sub[sub["domain"] == "signed"].set_index("seed")
        common = nn.index.intersection(sg.index)
        if len(common) < 2:
            continue
        a, b = nn.loc[common, "speedup"], sg.loc[common, "speedup"]
        t_stat, p_val = stats.ttest_rel(a, b)
        diff = a.values - b.values
        rows.append({
            "pair": pair,
            "n_seeds": len(common),
            "mean_speedup_nonnegative": a.mean(),
            "mean_speedup_signed": b.mean(),
            "mean_paired_diff (nonneg-signed)": diff.mean(),
            "std_paired_diff": diff.std(ddof=1) if len(diff) > 1 else float("nan"),
            "paired_t_p": p_val,
            "direction_reversed": bool((a.mean() > 1.0) != (b.mean() > 1.0)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Experiment B: fixed-target prerequisite-history matrix, both domains.
# --------------------------------------------------------------------------

def acquire_prior_skills_domain(prior_tasks, seed: int, condition_index: int, domain: str):
    skills = {}
    for task in prior_tasks:
        task_seed = _task_seed(seed, condition_index, task, offset=10_000)
        X, y = sample_task(task, N_TRAIN, seed=task_seed, domain=domain)
        net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=task_seed)
        net.reset_optimizer()
        train_to_accuracy(net, X, y)
        skills[task] = Skill(task, net, origin="scratch", parent=None)
    return skills


def run_history_condition_domain(target, prior_tasks, history_label, domain, seed, condition_index):
    skills = acquire_prior_skills_domain(prior_tasks, seed, condition_index, domain)

    controller_seed = _task_seed(seed, condition_index, target, offset=20_000)
    decision = comp.decide(skills, target, base_seed=controller_seed, domain=domain)

    target_train_seed = _task_seed(seed, condition_index, target, offset=30_000)
    X_target, y_target = sample_task(target, N_TRAIN, seed=target_train_seed, domain=domain)

    if decision["action"] == "reuse":
        steps, success = 0, True
        parent = decision["parent"]
        net = skills[parent].net
    elif decision["action"] == "clone":
        parent = decision["parent"]
        net = skills[parent].net.clone()
        net.reset_optimizer()
        steps, success = train_to_accuracy(net, X_target, y_target)
    else:
        parent = None
        net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=target_train_seed)
        net.reset_optimizer()
        steps, success = train_to_accuracy(net, X_target, y_target)

    test_seed = _task_seed(seed, condition_index, target, offset=40_000)
    X_test, y_test = sample_task(target, N_EVAL, seed=test_seed, domain=domain)

    return {
        "domain": domain,
        "target_task": target,
        "prior_history": history_label,
        "seed": seed,
        "strategy": decision["action"],
        "source_task": parent,
        "compatibility_score": float(decision["score"]),
        "adaptation_steps": steps,
        "acquisition_success": bool(success),
        "heldout_accuracy": float(net.accuracy(X_test, y_test, tol=ACC_TOL)),
        "heldout_mse": float(net.mse(X_test, y_test)),
    }


def run_all_history(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for condition_index, (target, prior_tasks, label) in enumerate(HISTORY_CONDITIONS):
            for seed in range(n_seeds):
                rows.append(run_history_condition_domain(target, prior_tasks, label, domain, seed, condition_index))
    return pd.DataFrame(rows)


def summarize_history(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, target, history), group in raw.groupby(["domain", "target_task", "prior_history"], sort=False):
        rows.append({
            "domain": domain,
            "target_task": target,
            "prior_history": history,
            "n_seeds": len(group),
            "acquisition_success_rate": group["acquisition_success"].mean(),
            "mean_adaptation_steps": group["adaptation_steps"].mean(),
            "std_adaptation_steps": group["adaptation_steps"].std(),
            "mean_heldout_accuracy": group["heldout_accuracy"].mean(),
            "reuse_count": int((group["strategy"] == "reuse").sum()),
            "clone_count": int((group["strategy"] == "clone").sum()),
            "scratch_count": int((group["strategy"] == "scratch").sum()),
        })
    order = {(t, h): i for i, (t, _, h) in enumerate(HISTORY_CONDITIONS)}
    summary = pd.DataFrame(rows)
    summary["_o1"] = summary["domain"].map({d: i for i, d in enumerate(DOMAINS)})
    summary["_o2"] = [order[(r.target_task, r.prior_history)] for r in summary.itertuples()]
    return summary.sort_values(["_o2", "_o1"]).drop(columns=["_o1", "_o2"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Sign-specific diagnostics (Section 10 of the plan).
# --------------------------------------------------------------------------

def _quadrant_labels(task: str, X: np.ndarray) -> np.ndarray:
    a = X[:, 0]
    if task in ("multiplication", "subtraction"):
        a_sign = np.where(a >= 0, "+", "-")
        b = X[:, 1]
        b_sign = np.where(b >= 0, "+", "-")
        return np.array([f"({sa},{sb})" for sa, sb in zip(a_sign, b_sign)])
    if task == "squares":
        return np.where(a >= 0, "positive base", "negative base")
    if task == "powers":
        exponent = np.round(X[:, 1] * 10).astype(int)
        base_sign = np.where(a >= 0, "positive base", "negative base")
        parity = np.where(exponent % 2 == 0, "even exponent", "odd exponent")
        labels = []
        for bs, p in zip(base_sign, parity):
            if bs == "positive base":
                labels.append("positive base")
            else:
                labels.append(f"negative base, {p}")
        return np.array(labels)
    raise ValueError(f"no quadrant scheme for task {task!r}")


def sign_breakdown(net, task: str, seed: int, n: int = 2000) -> pd.DataFrame:
    """Diagnostic-only: signed-domain accuracy/MSE of a trained network,
    broken down by the sign structure of its inputs. Not used to pick a
    stopping budget or threshold -- purely descriptive, computed after
    training on a fresh signed-domain batch."""
    X, y = sample_task(task, n, seed=seed, domain="signed")
    labels = _quadrant_labels(task, X)
    pred = net.predict(X)
    err = np.abs(pred - y)
    rows = []
    for label in sorted(set(labels)):
        mask = labels == label
        if mask.sum() == 0:
            continue
        rows.append({
            "task": task,
            "quadrant": label,
            "n": int(mask.sum()),
            "mean_abs_error": float(err[mask].mean()),
            "accuracy_tol_0.5": float((err[mask] <= ACC_TOL).mean()),
        })
    return pd.DataFrame(rows)


def run_sign_breakdowns(pairs_raw: pd.DataFrame, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """Re-trains the clone network for each diagnostic pair (signed domain
    only) and records its sign-quadrant breakdown, seed by seed."""
    rows = []
    for source, target in SIGN_DIAGNOSTIC_PAIRS:
        for seed in range(n_seeds):
            src_seed = _task_seed(seed, 1, source, offset=0)
            tgt_seed = _task_seed(seed, 1, target, offset=0)
            X_src, y_src = sample_task(source, N_TRAIN, seed=src_seed, domain="signed")
            parent = TinyMLP(hidden_dim=HIDDEN_DIM, seed=src_seed)
            parent.reset_optimizer()
            train_to_accuracy(parent, X_src, y_src)

            X_tgt, y_tgt = sample_task(target, N_TRAIN, seed=tgt_seed, domain="signed")
            clone = parent.clone()
            clone.reset_optimizer()
            train_to_accuracy(clone, X_tgt, y_tgt)

            breakdown = sign_breakdown(clone, target, seed=tgt_seed + 700_000)
            breakdown["pair"] = f"{source}->{target}"
            breakdown["seed"] = seed
            rows.append(breakdown)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def make_speedup_plot(pairs_summary: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [f"{s}->{t}" for s, t, _ in PAIRS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(pairs))
    for i, domain in enumerate(DOMAINS):
        means, stds = [], []
        for p in pairs:
            row = pairs_summary[(pairs_summary["domain"] == domain) & (pairs_summary["pair"] == p)]
            means.append(row["mean_speedup"].iloc[0] if len(row) else np.nan)
            stds.append(row["std_speedup"].iloc[0] if len(row) else np.nan)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3, label=domain)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15, ha="right")
    ax.set_ylabel("Mean paired speedup (scratch epochs / clone epochs)")
    ax.set_title("Transfer speedup: non-negative vs. signed domain\n(mean \u00b1 std over 15 seeds)")
    ax.legend(fontsize=8, title="Domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_success_rate_plot(history_summary: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = ["squares", "powers", "subtraction", "division"]
    fig, axes = plt.subplots(1, len(targets), figsize=(14, 4), sharey=True)
    condition_order = ["none", "addition", "addition+multiplication"]
    for ax, target in zip(axes, targets):
        x = np.arange(len(condition_order))
        width = 0.35
        for i, domain in enumerate(DOMAINS):
            vals = []
            for cond in condition_order:
                row = history_summary[
                    (history_summary["domain"] == domain)
                    & (history_summary["target_task"] == target)
                    & (history_summary["prior_history"] == cond)
                ]
                vals.append(row["acquisition_success_rate"].iloc[0] if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=domain)
        ax.set_xticks(x)
        ax.set_xticklabels(condition_order, rotation=30, ha="right", fontsize=7)
        ax.set_title(target.capitalize())
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Fixed-budget acquisition success rate")
    axes[-1].legend(fontsize=8, title="Domain")
    fig.suptitle("Acquisition reliability by prior-skill history: non-negative vs. signed domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_compatibility_vs_speedup_plot(pairs_raw: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"nonnegative": "o", "signed": "^"}
    for domain in DOMAINS:
        sub = pairs_raw[pairs_raw["domain"] == domain]
        grouped = sub.groupby("pair").agg(
            mean_score=("relatedness_score", "mean"), mean_speedup=("speedup", "mean")
        )
        ax.scatter(grouped["mean_score"], grouped["mean_speedup"], marker=markers[domain],
                   s=80, label=domain)
        for pair, row in grouped.iterrows():
            ax.annotate(f"{pair}\n({domain})", (row["mean_score"], row["mean_speedup"]),
                        fontsize=6, textcoords="offset points", xytext=(5, 5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean frozen-parent compatibility score")
    ax.set_ylabel("Mean paired speedup")
    ax.set_title("Compatibility score vs. transfer speedup, by domain")
    ax.legend(fontsize=8, title="Domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_convergence_plot(pairs_raw: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [f"{s}->{t}" for s, t, _ in PAIRS]
    fig, axes = plt.subplots(1, len(pairs), figsize=(16, 4), sharey=False)
    for ax, pair in zip(axes, pairs):
        sub = pairs_raw[pairs_raw["pair"] == pair]
        data, labels = [], []
        for domain in DOMAINS:
            for col, name in [("clone_epochs", "clone"), ("scratch_epochs", "scratch")]:
                data.append(sub[sub["domain"] == domain][col].values)
                labels.append(f"{domain}\n{name}")
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(pair, fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_ylabel("Epochs to 85% training accuracy")
    fig.suptitle("Convergence distributions: clone vs. scratch, by domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    print("Running relatedness-pair experiment (A) across both domains...")
    pairs_raw = run_all_pairs()
    pairs_summary = summarize_pairs(pairs_raw)
    pairs_domain_comparison = paired_domain_comparison_pairs(pairs_raw)

    print("Running fixed-target history experiment (B) across both domains...")
    history_raw = run_all_history()
    history_summary = summarize_history(history_raw)

    print("Running sign-specific diagnostics...")
    sign_breakdown_raw = run_sign_breakdowns(pairs_raw)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    pairs_raw.to_csv(out_dir / "signed_domain_pairs.csv", index=False)
    pairs_summary.to_csv(out_dir / "signed_domain_pairs_summary.csv", index=False)
    pairs_domain_comparison.to_csv(out_dir / "signed_domain_pairs_comparison.csv", index=False)
    history_raw.to_csv(out_dir / "signed_domain_history.csv", index=False)
    history_summary.to_csv(out_dir / "signed_domain_history_summary.csv", index=False)
    sign_breakdown_raw.to_csv(out_dir / "signed_domain_sign_breakdown.csv", index=False)

    make_speedup_plot(pairs_summary, out_dir / "plot_signed_domain_speedup.png")
    make_success_rate_plot(history_summary, out_dir / "plot_signed_domain_success_rate.png")
    make_compatibility_vs_speedup_plot(pairs_raw, out_dir / "plot_signed_domain_compatibility_vs_speedup.png")
    make_convergence_plot(pairs_raw, out_dir / "plot_signed_domain_convergence.png")

    print()
    print("=== Pairs summary ===")
    print(pairs_summary.to_string(index=False))
    print()
    print("=== Domain comparison (paired by seed) ===")
    print(pairs_domain_comparison.to_string(index=False))
    print()
    print("=== History summary ===")
    print(history_summary.drop(columns=["std_adaptation_steps"]).to_string(index=False))
    print()
    print("=== Sign breakdown (aggregated preview) ===")
    if len(sign_breakdown_raw):
        print(sign_breakdown_raw.groupby(["pair", "quadrant"])[["mean_abs_error", "accuracy_tol_0.5"]]
              .mean().to_string())


if __name__ == "__main__":
    main()
