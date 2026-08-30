"""Signed-domain transfer-robustness / domain-sensitivity analysis.

The experiment compares the existing non-negative arithmetic domain with a
matched signed-integer domain. A prerequisite/source skill is only exposed to
the controller after it has actually reached the declared acquisition
criterion. Likewise, the forced-clone pair analysis is only a valid transfer
comparison when the source skill was successfully acquired.
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

SIGN_DIAGNOSTIC_PAIRS = [
    ("multiplication", "powers"),
    ("multiplication", "squares"),
    ("addition", "subtraction"),
]


def train_to_accuracy(net, X, y, max_epochs: int = MAX_EPOCHS):
    """Train up to max_epochs and return (epochs_taken, reached_target)."""
    for epoch in range(1, max_epochs + 1):
        net.train_step(X, y, lr=LR)
        if net.accuracy(X, y, tol=ACC_TOL) >= ACC_TARGET:
            return epoch, True
    return max_epochs, False


def _task_seed(seed: int, salt: int, task: str, offset: int = 0) -> int:
    return seed * 100_000 + salt * 1_000 + TASK_SEED_INDEX[task] + offset


def _invalid_pair_row(source, target, relatedness_label, domain, seed, parent_epochs):
    """Return an explicit invalid comparison when source acquisition fails."""
    return {
        "domain": domain,
        "pair": f"{source}->{target}",
        "relatedness_label": relatedness_label,
        "source_task": source,
        "target_task": target,
        "seed": seed,
        "relatedness_score": np.nan,
        "solve_accuracy": np.nan,
        "parent_epochs": parent_epochs,
        "parent_reached": False,
        "pair_valid": False,
        "clone_epochs": np.nan,
        "clone_success": np.nan,
        "scratch_epochs": np.nan,
        "scratch_success": np.nan,
        "speedup": np.nan,
        "clone_heldout_mse": np.nan,
        "scratch_heldout_mse": np.nan,
    }


def run_pair_domain(source: str, target: str, relatedness_label: str, domain: str, seed: int) -> dict:
    """Run a forced-clone comparison only after source acquisition succeeds."""
    src_seed = _task_seed(seed, 1, source)
    tgt_seed = _task_seed(seed, 1, target)

    X_src, y_src = sample_task(source, N_TRAIN, seed=src_seed, domain=domain)
    parent = TinyMLP(hidden_dim=HIDDEN_DIM, seed=src_seed)
    parent.reset_optimizer()
    parent_epochs, parent_reached = train_to_accuracy(parent, X_src, y_src)
    if not parent_reached:
        return _invalid_pair_row(source, target, relatedness_label, domain, seed, parent_epochs)

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
        "parent_reached": True,
        "pair_valid": True,
        "clone_epochs": clone_epochs,
        "clone_success": clone_success,
        "scratch_epochs": scratch_epochs,
        "scratch_success": scratch_success,
        "speedup": scratch_epochs / clone_epochs,
        "clone_heldout_mse": float(clone.mse(X_hold, y_hold)),
        "scratch_heldout_mse": float(scratch.mse(X_hold, y_hold)),
    }


def run_all_pairs(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    return pd.DataFrame(
        run_pair_domain(source, target, label, domain, seed)
        for domain in DOMAINS
        for source, target, label in PAIRS
        for seed in range(n_seeds)
    )


def summarize_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, pair), group in raw.groupby(["domain", "pair"], sort=False):
        valid = group[group["pair_valid"]]
        rows.append({
            "domain": domain,
            "pair": pair,
            "relatedness_label": group["relatedness_label"].iloc[0],
            "n_seeds": len(group),
            "valid_pairs": len(valid),
            "source_failures": int((~group["parent_reached"]).sum()),
            "mean_relatedness_score": valid["relatedness_score"].mean() if len(valid) else np.nan,
            "mean_clone_epochs": valid["clone_epochs"].mean() if len(valid) else np.nan,
            "std_clone_epochs": valid["clone_epochs"].std() if len(valid) else np.nan,
            "mean_scratch_epochs": valid["scratch_epochs"].mean() if len(valid) else np.nan,
            "std_scratch_epochs": valid["scratch_epochs"].std() if len(valid) else np.nan,
            "clone_success_rate": valid["clone_success"].mean() if len(valid) else np.nan,
            "scratch_success_rate": valid["scratch_success"].mean() if len(valid) else np.nan,
            "mean_speedup": valid["speedup"].mean() if len(valid) else np.nan,
            "std_speedup": valid["speedup"].std() if len(valid) else np.nan,
            "median_speedup": valid["speedup"].median() if len(valid) else np.nan,
        })
    order = {f"{s}->{t}": i for i, (s, t, _) in enumerate(PAIRS)}
    summary = pd.DataFrame(rows)
    summary["_o1"] = summary["domain"].map({d: i for i, d in enumerate(DOMAINS)})
    summary["_o2"] = summary["pair"].map(order)
    return summary.sort_values(["_o2", "_o1"]).drop(columns=["_o1", "_o2"]).reset_index(drop=True)


def paired_domain_comparison_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare domain speedups only for seeds valid in both domains."""
    rows = []
    for pair in raw["pair"].unique():
        sub = raw[raw["pair"] == pair]
        nn = sub[(sub["domain"] == "nonnegative") & sub["pair_valid"]].set_index("seed")
        sg = sub[(sub["domain"] == "signed") & sub["pair_valid"]].set_index("seed")
        common = nn.index.intersection(sg.index)
        if len(common) < 2:
            continue
        a = nn.loc[common, "speedup"]
        b = sg.loc[common, "speedup"]
        _, p_val = stats.ttest_rel(a, b)
        diff = a.values - b.values
        rows.append({
            "pair": pair,
            "n_seeds": len(common),
            "mean_speedup_nonnegative": a.mean(),
            "mean_speedup_signed": b.mean(),
            "mean_paired_diff (nonneg-signed)": diff.mean(),
            "std_paired_diff": diff.std(ddof=1),
            "paired_t_p": p_val,
            "direction_reversed": bool((a.mean() > 1.0) != (b.mean() > 1.0)),
        })
    return pd.DataFrame(rows)


def acquire_prior_skills_domain(prior_tasks, seed: int, condition_index: int, domain: str):
    """Return skills plus per-prerequisite outcomes; failed prerequisites are unavailable."""
    skills = {}
    prior_rows = []
    history_valid = True
    for task in prior_tasks:
        task_seed = _task_seed(seed, condition_index, task, offset=10_000)
        X, y = sample_task(task, N_TRAIN, seed=task_seed, domain=domain)
        net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=task_seed)
        net.reset_optimizer()
        steps, success = train_to_accuracy(net, X, y)
        prior_rows.append({
            "prior_task": task,
            "prior_training_steps": steps,
            "prior_acquisition_success": bool(success),
        })
        if success:
            skills[task] = Skill(task, net, origin="scratch", parent=None)
        else:
            history_valid = False
    return skills, prior_rows, history_valid


def run_history_condition_domain(target, prior_tasks, history_label, domain, seed, condition_index):
    skills, prior_rows, history_valid = acquire_prior_skills_domain(
        prior_tasks, seed, condition_index, domain
    )
    row = {
        "domain": domain,
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
    row.update({
        "strategy": decision["action"],
        "source_task": parent,
        "compatibility_score": float(decision["score"]),
        "solve_accuracy": float(decision["solve_accuracy"]),
        "adaptation_steps": steps,
        "fixed_budget": MAX_EPOCHS,
        "target_attempted": True,
        "acquisition_success": bool(success),
        "heldout_accuracy": float(net.accuracy(X_test, y_test, tol=ACC_TOL)),
        "heldout_mse": float(net.mse(X_test, y_test)),
    })
    return row


def run_all_history(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    return pd.DataFrame(
        run_history_condition_domain(target, prior, label, domain, seed, i)
        for domain in DOMAINS
        for i, (target, prior, label) in enumerate(HISTORY_CONDITIONS)
        for seed in range(n_seeds)
    )


def summarize_history(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, target, history), group in raw.groupby(
        ["domain", "target_task", "prior_history"], sort=False
    ):
        attempted = group[group["target_attempted"]]
        rows.append({
            "domain": domain,
            "target_task": target,
            "prior_history": history,
            "n_seeds": len(group),
            "prior_history_valid": int(group["prior_history_valid"].sum()),
            "prerequisite_failures": int((~group["prior_history_valid"]).sum()),
            "target_attempts": int(group["target_attempted"].sum()),
            "acquisition_successes": int(group["acquisition_success"].sum()),
            "acquisition_success_rate": attempted["acquisition_success"].mean() if len(attempted) else np.nan,
            "mean_adaptation_steps": attempted["adaptation_steps"].mean() if len(attempted) else np.nan,
            "std_adaptation_steps": attempted["adaptation_steps"].std() if len(attempted) else np.nan,
            "mean_heldout_accuracy": attempted["heldout_accuracy"].mean() if len(attempted) else np.nan,
            "std_heldout_accuracy": attempted["heldout_accuracy"].std() if len(attempted) else np.nan,
            "reuse_count": int((group["strategy"] == "reuse").sum()),
            "clone_count": int((group["strategy"] == "clone").sum()),
            "scratch_count": int((group["strategy"] == "scratch").sum()),
            "prerequisite_failed_count": int((group["strategy"] == "prerequisite_failed").sum()),
        })
    order = {(t, h): i for i, (t, _, h) in enumerate(HISTORY_CONDITIONS)}
    summary = pd.DataFrame(rows)
    summary["_o1"] = summary["domain"].map({d: i for i, d in enumerate(DOMAINS)})
    summary["_o2"] = [order[(r.target_task, r.prior_history)] for r in summary.itertuples()]
    return summary.sort_values(["_o2", "_o1"]).drop(columns=["_o1", "_o2"]).reset_index(drop=True)


def _quadrant_labels(task: str, X: np.ndarray) -> np.ndarray:
    a = X[:, 0]
    if task in ("multiplication", "subtraction"):
        b = X[:, 1]
        sa = np.where(a >= 0, "+", "-")
        sb = np.where(b >= 0, "+", "-")
        return np.array([f"({x},{y})" for x, y in zip(sa, sb)])
    if task == "squares":
        return np.where(a >= 0, "positive base", "negative base")
    if task == "powers":
        exponent = np.round(X[:, 1] * 10).astype(int)
        base_sign = np.where(a >= 0, "positive base", "negative base")
        parity = np.where(exponent % 2 == 0, "even exponent", "odd exponent")
        return np.array([
            "positive base" if bs == "positive base" else f"negative base, {p}"
            for bs, p in zip(base_sign, parity)
        ])
    raise ValueError(f"no quadrant scheme for task {task!r}")


def sign_breakdown(net, task: str, seed: int, n: int = 2000) -> pd.DataFrame:
    X, y = sample_task(task, n, seed=seed, domain="signed")
    labels = _quadrant_labels(task, X)
    err = np.abs(net.predict(X) - y)
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


def run_sign_breakdowns(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    rows = []
    for source, target in SIGN_DIAGNOSTIC_PAIRS:
        for seed in range(n_seeds):
            src_seed = _task_seed(seed, 1, source)
            tgt_seed = _task_seed(seed, 1, target)
            X_src, y_src = sample_task(source, N_TRAIN, seed=src_seed, domain="signed")
            parent = TinyMLP(hidden_dim=HIDDEN_DIM, seed=src_seed)
            parent.reset_optimizer()
            _, reached = train_to_accuracy(parent, X_src, y_src)
            if not reached:
                continue
            X_tgt, y_tgt = sample_task(target, N_TRAIN, seed=tgt_seed, domain="signed")
            clone = parent.clone()
            clone.reset_optimizer()
            train_to_accuracy(clone, X_tgt, y_tgt)
            breakdown = sign_breakdown(clone, target, seed=tgt_seed + 700_000)
            breakdown["pair"] = f"{source}->{target}"
            breakdown["seed"] = seed
            rows.append(breakdown)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_speedup_plot(summary: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [f"{s}->{t}" for s, t, _ in PAIRS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(pairs))
    for i, domain in enumerate(DOMAINS):
        sub = summary[summary["domain"] == domain].set_index("pair")
        means = [sub.loc[p, "mean_speedup"] if p in sub.index else np.nan for p in pairs]
        stds = [sub.loc[p, "std_speedup"] if p in sub.index else np.nan for p in pairs]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3, label=domain)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15, ha="right")
    ax.set_ylabel("Mean paired speedup (scratch epochs / clone epochs)")
    ax.set_title("Transfer speedup: non-negative vs. signed domain")
    ax.legend(fontsize=8, title="Domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_success_rate_plot(summary: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = ["squares", "powers", "subtraction", "division"]
    conditions = ["none", "addition", "addition+multiplication"]
    fig, axes = plt.subplots(1, len(targets), figsize=(14, 4), sharey=True)
    for ax, target in zip(axes, targets):
        x = np.arange(len(conditions))
        width = 0.35
        for i, domain in enumerate(DOMAINS):
            sub = summary[(summary["domain"] == domain) & (summary["target_task"] == target)].set_index("prior_history")
            vals = [sub.loc[c, "acquisition_success_rate"] if c in sub.index else np.nan for c in conditions]
            ax.bar(x + (i - 0.5) * width, vals, width, label=domain)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=7)
        ax.set_title(target.capitalize())
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Fixed-budget acquisition success rate (valid histories)")
    axes[-1].legend(fontsize=8, title="Domain")
    fig.suptitle("Acquisition reliability by prior-skill history")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_compatibility_vs_speedup_plot(raw: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for domain, marker in (("nonnegative", "o"), ("signed", "^")):
        sub = raw[(raw["domain"] == domain) & raw["pair_valid"]]
        grouped = sub.groupby("pair").agg(mean_score=("relatedness_score", "mean"), mean_speedup=("speedup", "mean"))
        ax.scatter(grouped["mean_score"], grouped["mean_speedup"], marker=marker, s=80, label=domain)
        for pair, row in grouped.iterrows():
            ax.annotate(f"{pair}\n({domain})", (row["mean_score"], row["mean_speedup"]), fontsize=6, xytext=(5, 5), textcoords="offset points")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean frozen-parent compatibility score")
    ax.set_ylabel("Mean paired speedup")
    ax.set_title("Compatibility score vs. transfer speedup, by domain")
    ax.legend(fontsize=8, title="Domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_convergence_plot(raw: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [f"{s}->{t}" for s, t, _ in PAIRS]
    fig, axes = plt.subplots(1, len(pairs), figsize=(16, 4), sharey=False)
    for ax, pair in zip(axes, pairs):
        sub = raw[(raw["pair"] == pair) & raw["pair_valid"]]
        data, labels = [], []
        for domain in DOMAINS:
            for col, name in (("clone_epochs", "clone"), ("scratch_epochs", "scratch")):
                data.append(sub[sub["domain"] == domain][col].dropna().values)
                labels.append(f"{domain}\n{name}")
        if all(len(v) for v in data):
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
    sign_breakdown_raw = run_sign_breakdowns()

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

    print("\n=== Pairs summary ===")
    print(pairs_summary.to_string(index=False))
    print("\n=== Domain comparison (paired by seed) ===")
    print(pairs_domain_comparison.to_string(index=False))
    print("\n=== History summary ===")
    print(history_summary.drop(columns=["std_adaptation_steps"]).to_string(index=False))


if __name__ == "__main__":
    main()
