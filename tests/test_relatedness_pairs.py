"""Structural tests for the prerequisite-based acquisition experiment.

These tests deliberately avoid asserting the empirical direction of transfer.
The experiment must be free to find benefit, no benefit, or negative transfer.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import pandas as pd  # noqa: E402
from relatedness_pairs import CONDITIONS, MAX_EPOCHS, run_condition, summarize  # noqa: E402


class RelatednessPairsTests(unittest.TestCase):
    def test_condition_matrix_has_fixed_targets_and_three_histories(self):
        self.assertEqual(len(CONDITIONS), 12)
        for target in ("subtraction", "division", "squares", "powers"):
            histories = [label for t, _, label in CONDITIONS if t == target]
            self.assertEqual(histories, ["none", "addition", "addition+multiplication"])

    def test_scratch_condition_has_no_prior_skills(self):
        row = run_condition("division", (), "none", seed=0, condition_index=0)
        self.assertEqual(row["strategy"], "scratch")
        self.assertIsNone(row["source_task"])
        self.assertEqual(row["prior_tasks"], "none")
        self.assertGreaterEqual(row["adaptation_steps"], 1)
        self.assertLessEqual(row["adaptation_steps"], MAX_EPOCHS)
        self.assertIn("acquisition_success", row)
        self.assertIn("heldout_accuracy", row)
        self.assertIn("heldout_mse", row)

    def test_prior_history_records_actual_acquired_skills(self):
        row = run_condition(
            "division", ("addition", "multiplication"),
            "addition+multiplication", seed=0, condition_index=5,
        )
        self.assertEqual(row["prior_tasks"], "addition+multiplication")
        self.assertIn("addition_prior_steps", row)
        self.assertIn("multiplication_prior_steps", row)
        self.assertTrue(row["addition_prior_success"])
        self.assertTrue(row["multiplication_prior_success"])
        self.assertIn(row["strategy"], {"reuse", "clone", "scratch"})
        self.assertGreaterEqual(row["compatibility_score"], 0.0)
        self.assertLessEqual(row["compatibility_score"], 1.0)

    def test_summarize_preserves_condition_order(self):
        rows = [
            run_condition(target, prior, label, seed=0, condition_index=i)
            for i, (target, prior, label) in enumerate(CONDITIONS)
        ]
        summary = summarize(pd.DataFrame(rows))
        expected = [(target, label) for target, _, label in CONDITIONS]
        actual = list(zip(summary["target_task"], summary["prior_history"]))
        self.assertEqual(actual, expected)
        self.assertEqual(len(summary), len(CONDITIONS))


if __name__ == "__main__":
    unittest.main()
