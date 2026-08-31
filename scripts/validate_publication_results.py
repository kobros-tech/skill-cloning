"""Validate that publication-facing result sources exist and agree on key invariants."""

from __future__ import annotations

import csv
from pathlib import Path

R = Path("results")

REQUIRED = [
    "relatedness_pairs.csv",
    "relatedness_pairs_summary.csv",
    "signed_domain_pairs.csv",
    "signed_domain_pairs_summary.csv",
    "signed_domain_pairs_comparison.csv",
    "signed_domain_history.csv",
    "signed_domain_sign_breakdown.csv",
    "retention.csv",
    "retention_summary.csv",
    "compatibility_calibration.csv",
    "compatibility_calibration_summary.csv",
    "unrelated_parent_control_stats.csv",
    "stopping_rule_confound_summary.csv",
]


def rows(name: str):
    with (R / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    missing = [name for name in REQUIRED if not (R / name).exists()]
    if missing:
        raise SystemExit("Missing workflow result files: " + ", ".join(missing))

    fixed = rows("relatedness_pairs_summary.csv")
    expected = {
        ("subtraction", "none"),
        ("subtraction", "addition"),
        ("subtraction", "addition+multiplication"),
        ("division", "none"),
        ("division", "addition"),
        ("division", "addition+multiplication"),
        ("squares", "none"),
        ("squares", "addition"),
        ("squares", "addition+multiplication"),
        ("powers", "none"),
        ("powers", "addition"),
        ("powers", "addition+multiplication"),
    }
    observed = {(r["target_task"], r["prior_history"]) for r in fixed}
    if observed != expected:
        raise SystemExit(f"Fixed-target condition mismatch: {sorted(observed)}")

    # Regression checks for the values that are most likely to become stale in manuscript tables.
    div = next(r for r in fixed if r["target_task"] == "division" and r["prior_history"] == "addition+multiplication")
    if abs(float(div["mean_adaptation_steps"]) - 621.7857142857143) > 1e-9:
        raise SystemExit("Division final-history mean no longer matches the authoritative workflow value")
    if int(div["prior_history_valid"]) != 14 or int(div["prerequisite_failures"]) != 1:
        raise SystemExit("Division final-history validity/failure counts changed unexpectedly")

    signed = rows("signed_domain_pairs_comparison.csv")
    if len(signed) != 4:
        raise SystemExit("Signed-domain comparison must contain exactly four planned pairs")
    powers = next(r for r in signed if r["pair"] == "multiplication->powers")
    if int(powers["n_seeds"]) != 5 or r_bool(powers["direction_reversed"]) is not True:
        raise SystemExit("Signed multiplication->powers comparison does not match the current workflow structure")

    print("Publication result validation passed.")


def r_bool(value: str) -> bool:
    return value.strip().lower() == "true"


if __name__ == "__main__":
    main()
