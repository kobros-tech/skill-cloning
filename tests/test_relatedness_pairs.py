```python
"""Structural tests for the prerequisite-based acquisition experiment.

These tests deliberately avoid asserting the empirical direction of transfer.
The experiment must be free to find benefit, no benefit, or negative transfer.

The tests cover:
- the fixed-target prerequisite matrix;
- scratch acquisition with no prior skills;
- recording of actually acquired prerequisite skills;
- fail-closed behavior when a prerequisite cannot be acquired;
- preservation of condition ordering;
- explicit domain propagation for the signed-domain follow-up.
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
from relatedness_pairs import (  # noqa: E402
    CONDITIONS,
    MAX_EPOCHS,
    run_condition,
    summarize,
)


class RelatednessPairsTests(unittest.TestCase):
    def test_condition_matrix_has_fixed_targets_and_three_histories(self):
        self.assertEqual(len(CONDITIONS), 12)

        for target in ("subtraction", "division", "squares", "powers"):
            histories = [
                label
                for t, _, label in CONDITIONS
                if t == target
            ]
            self.assertEqual(
                histories,
                ["none", "addition", "addition+multiplication"],
            )

    def test_scratch_condition_has_no_prior_skills(self):
        row = run_condition(
            "division",
            (),
            "none",
            seed=0,
            condition_index=0,
        )

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
            "division",
            ("addition", "multiplication"),
            "addition+multiplication",
            seed=0,
            condition_index=5,
        )

        self.assertEqual(
            row["prior_tasks"],
            "addition+multiplication",
        )
        self.assertTrue(row["prior_history_valid"])

        self.assertIn("addition_prior_steps", row)
        self.assertIn("multiplication_prior_steps", row)

        self.assertTrue(row["addition_prior_success"])
        self.assertTrue(row["multiplication_prior_success"])

        # The controller is allowed to discover any of the three routes.
        self.assertIn(
            row["strategy"],
            {"reuse", "clone", "scratch"},
        )

        self.assertGreaterEqual(
            row["compatibility_score"],
            0.0,
        )
        self.assertLessEqual(
            row["compatibility_score"],
            1.0,
        )

    def test_failed_prerequisite_is_not_exposed_to_controller(self):
        def fail_second_prerequisite(net, X, y):
            # First prerequisite succeeds; second fails at the fixed budget.
            if not hasattr(fail_second_prerequisite, "calls"):
                fail_second_prerequisite.calls = 0

            fail_second_prerequisite.calls += 1

            if fail_second_prerequisite.calls == 1:
                return 1, True

            return MAX_EPOCHS, False

        with (
            patch(
                "relatedness_pairs.train_to_accuracy",
                side_effect=fail_second_prerequisite,
            ),
            patch("relatedness_pairs.comp.decide") as decide,
        ):
            row = run_condition(
                "division",
                ("addition", "multiplication"),
                "addition+multiplication",
                seed=0,
                condition_index=5,
            )

        # A failed requested history must fail closed rather than being
        # silently downgraded to a partial history.
        decide.assert_not_called()

        self.assertFalse(row["prior_history_valid"])
        self.assertTrue(row["addition_prior_success"])
        self.assertFalse(row["multiplication_prior_success"])

        self.assertEqual(
            row["strategy"],
            "prerequisite_failed",
        )
        self.assertFalse(row["target_attempted"])
        self.assertFalse(row["acquisition_success"])

        self.assertIsNone(row["source_task"])
        self.assertTrue(pd.isna(row["adaptation_steps"]))
        self.assertTrue(pd.isna(row["heldout_mse"]))

    def test_summarize_preserves_condition_order(self):
        rows = [
            run_condition(
                target,
                prior,
                label,
                seed=0,
                condition_index=i,
            )
            for i, (target, prior, label)
            in enumerate(CONDITIONS)
        ]

        summary = summarize(pd.DataFrame(rows))

        expected = [
            (target, label)
            for target, _, label in CONDITIONS
        ]
        actual = list(
            zip(
                summary["target_task"],
                summary["prior_history"],
            )
        )

        self.assertEqual(actual, expected)
        self.assertEqual(len(summary), len(CONDITIONS))

    def test_domain_is_forwarded_to_task_sampling(self):
        """The signed-domain experiment must not silently use the default domain."""
        with patch(
            "relatedness_pairs.sample_task",
            side_effect=lambda task, n, seed, domain="nonnegative": (
                # Keep the mock lightweight while preserving the expected
                # shape and dtype used by the training code.
                __import__("numpy").zeros((n, 2), dtype=float),
                __import__("numpy").zeros(n, dtype=float),
            ),
        ) as sample_task:
            # This test only verifies the API contract. The actual signed
            # domain behavior is tested in tasks.py tests.
            try:
                run_condition(
                    "division",
                    (),
                    "none",
                    seed=0,
                    condition_index=0,
                    domain="signed",
                )
            except TypeError as exc:
                self.fail(
                    "run_condition() does not expose the required "
                    "'domain' argument for the signed-domain experiment: "
                    f"{exc}"
                )

        # At least the target training/evaluation calls must receive the
        # requested signed domain.
        domains = [
            call.kwargs.get("domain")
            for call in sample_task.call_args_list
        ]
        self.assertTrue(domains)
        self.assertTrue(
            all(domain == "signed" for domain in domains)
        )


if __name__ == "__main__":
    unittest.main()
```
