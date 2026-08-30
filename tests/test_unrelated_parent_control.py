"""Tests for the matched relevant/unrelated/scratch parent control."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from unrelated_parent_control import paired_summary  # noqa: E402


class UnrelatedParentControlTests(unittest.TestCase):
    def test_paired_summary_uses_matched_seed_unit(self):
        rows = []
        for seed in range(3):
            for arm, steps in {
                "relevant_clone": 10 + seed,
                "unrelated_clone": 20 + seed,
                "scratch": 30 + seed,
            }.items():
                rows.append({"seed": seed, "target": "powers", "arm": arm, "convergence_steps": steps})

        summary = paired_summary(pd.DataFrame(rows))
        self.assertEqual(set(summary["comparison"]), {
            "relevant_minus_scratch",
            "relevant_minus_unrelated",
            "unrelated_minus_scratch",
        })
        self.assertTrue((summary["n_matched"] == 3).all())

    def test_paired_summary_reports_expected_direction(self):
        rows = []
        for seed in range(4):
            rows.extend([
                {"seed": seed, "target": "multiplication", "arm": "relevant_clone", "convergence_steps": 10 + seed},
                {"seed": seed, "target": "multiplication", "arm": "unrelated_clone", "convergence_steps": 20 + seed},
                {"seed": seed, "target": "multiplication", "arm": "scratch", "convergence_steps": 30 + seed},
            ])

        summary = paired_summary(pd.DataFrame(rows)).set_index("comparison")
        self.assertLess(summary.loc["relevant_minus_scratch", "mean_difference_steps"], 0)
        self.assertLess(summary.loc["relevant_minus_unrelated", "mean_difference_steps"], 0)
        self.assertLess(summary.loc["unrelated_minus_scratch", "mean_difference_steps"], 0)


if __name__ == "__main__":
    unittest.main()
