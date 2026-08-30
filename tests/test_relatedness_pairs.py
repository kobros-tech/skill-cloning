"""Structural tests for the prerequisite-based acquisition experiment.

These tests deliberately avoid asserting the empirical direction of transfer.
The experiment must be free to find benefit, no benefit, or negative transfer.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

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
        self.assertTrue(row["prior_history_valid"])
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
        self.assertTrue(row["prior_history_valid"])
        self.assertIn("addition_prior_steps", row)
        self.assertIn("multiplication_prior_steps", row)
        self.assertTrue(row["addition_prior_success"])
        self.assertTrue(row["multiplication_prior_success"])
        self.assertIn(row["strategy"], {"reuse", "clone", "scratch"})
        self.assertGreaterEqual(row["compatibility_score"], 0.0)
        self.assertLessEqual(row["compatibility_score"], 1.0)

    def test_failed_prerequisite_is_not_exposed_to_controller(self):
        def fail_second_prerequisite(net, X, y):
            # First prerequisite succeeds; second fails at the fixed budget.
            if not hasattr(fail_second_prerequisite, "calls"):
                fail_second_prerequisite.calls = 0
            fail_second_prerequisite.calls += 1
            return (1, True) if fail_second_prerequisite.calls == 1 else (MAX_EPOCHS, False)

        with patch("relatedness_pairs.train_to_accuracy", side_effect=fail_second_prerequisite), \
             patch("relatedness_pairs.comp.decide") as decide:
            row = run_condition(
                "division", ("addition", "multiplication"),
                "addition+multiplication", seed=0, condition_index=5,
            )

        decide.assert_not_called()
        self.assertFalse(row["prior_history_valid"])
        self.assertTrue(row["addition_prior_success"])
        self.assertFalse(row["multiplication_prior_success"])
        self.assertEqual(row["strategy"], "prerequisite_failed")
        self.assertFalse(row["target_attempted"])
        self.assertFalse(row["acquisition_success"])
        self.assertIsNone(row["source_task"])
        self.assertTrue(pd.isna(row["adaptation_steps"]))
        self.assertTrue(pd.isna(row["heldout_mse"]))

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
