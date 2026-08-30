"""Regression tests for signed-domain sampling and acquisition semantics."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
from tasks import DOMAINS, sample_task  # noqa: E402


class DomainConfigurationTests(unittest.TestCase):
    def test_nonnegative_domain_never_produces_negative_operands(self):
        for task in ["addition", "subtraction", "multiplication", "powers", "squares", "division"]:
            X, _ = sample_task(task, 500, seed=1, domain="nonnegative")
            self.assertTrue((X >= 0).all())

    def test_signed_domain_produces_negative_operands(self):
        for task in ["addition", "subtraction", "multiplication", "squares", "division"]:
            X, _ = sample_task(task, 500, seed=1, domain="signed")
            self.assertTrue((X < 0).any())

    def test_signed_powers_base_can_be_negative_but_exponent_cannot(self):
        X, _ = sample_task("powers", 2000, seed=1, domain="signed")
        self.assertTrue((X[:, 0] < 0).any())
        self.assertTrue((X[:, 1] >= 0).all())

    def test_division_never_samples_a_zero_divisor(self):
        for domain in DOMAINS:
            X, y = sample_task("division", 2000, seed=1, domain=domain)
            self.assertTrue((X[:, 1] != 0).all())
            self.assertTrue(np.isfinite(y).all())

    def test_signed_division_divisor_covers_both_signs(self):
        X, _ = sample_task("division", 2000, seed=1, domain="signed")
        self.assertTrue((X[:, 1] > 0).any())
        self.assertTrue((X[:, 1] < 0).any())

    def test_input_scaling_is_consistent_across_domains(self):
        for domain in DOMAINS:
            for task in ["addition", "multiplication", "squares"]:
                X, _ = sample_task(task, 200, seed=3, domain=domain)
                raw = X * 10.0
                self.assertTrue(np.allclose(raw, np.round(raw)))

    def test_sample_count_is_controlled(self):
        for domain in DOMAINS:
            X, y = sample_task("addition", 137, seed=1, domain=domain)
            self.assertEqual(X.shape[0], 137)
            self.assertEqual(y.shape[0], 137)

    def test_both_domains_are_reproducible_under_a_fixed_seed(self):
        for domain in DOMAINS:
            X1, y1 = sample_task("multiplication", 100, seed=7, domain=domain)
            X2, y2 = sample_task("multiplication", 100, seed=7, domain=domain)
            self.assertTrue(np.array_equal(X1, X2))
            self.assertTrue(np.array_equal(y1, y2))

    def test_nonnegative_domain_matches_original_division_sequence(self):
        X, y = sample_task("division", 10, seed=42, domain="nonnegative")
        rng = np.random.default_rng(42)
        a = rng.integers(0, 25, size=10)
        b = rng.integers(1, 6, size=10)
        expected_y = a.astype(float) / b.astype(float)
        expected_X = np.stack([a, b], axis=1).astype(float) / 10.0
        self.assertTrue(np.array_equal(X, expected_X))
        self.assertTrue(np.array_equal(y, expected_y))


class CompatibilityDomainThreadingTests(unittest.TestCase):
    def test_compatibility_score_uses_the_requested_domain(self):
        import compatibility as comp
        from skill import TinyMLP

        net = TinyMLP(hidden_dim=8, seed=1)
        score_nonneg = comp.compatibility_score(net, "multiplication", 1, domain="nonnegative")
        score_signed = comp.compatibility_score(net, "multiplication", 1, domain="signed")
        self.assertNotEqual(score_nonneg, score_signed)

    def test_decide_defaults_to_nonnegative_domain(self):
        import inspect
        import compatibility as comp
        sig = inspect.signature(comp.decide)
        self.assertEqual(sig.parameters["domain"].default, "nonnegative")


class AcquisitionFailureSemanticsTests(unittest.TestCase):
    def test_failed_prerequisite_is_not_exposed_to_controller(self):
        from relatedness_pairs import MAX_EPOCHS, run_condition

        def fail_second(net, X, y):
            if not hasattr(fail_second, "calls"):
                fail_second.calls = 0
            fail_second.calls += 1
            return (1, True) if fail_second.calls == 1 else (MAX_EPOCHS, False)

        with patch("relatedness_pairs.train_to_accuracy", side_effect=fail_second), \
             patch("relatedness_pairs.comp.decide") as decide:
            row = run_condition(
                "division", ("addition", "multiplication"),
                "addition+multiplication", seed=0, condition_index=5,
            )

        decide.assert_not_called()
        self.assertFalse(row["prior_history_valid"])
        self.assertEqual(row["strategy"], "prerequisite_failed")
        self.assertFalse(row["target_attempted"])
        self.assertFalse(row["acquisition_success"])
        self.assertFalse(row["multiplication_prior_success"])
        self.assertIsNone(row["source_task"])

    def test_signed_history_does_not_expose_failed_prerequisite(self):
        from signed_domain_transfer import run_history_condition_domain, MAX_EPOCHS

        def fail_second(net, X, y):
            if not hasattr(fail_second, "calls"):
                fail_second.calls = 0
            fail_second.calls += 1
            return (1, True) if fail_second.calls == 1 else (MAX_EPOCHS, False)

        with patch("signed_domain_transfer.train_to_accuracy", side_effect=fail_second), \
             patch("signed_domain_transfer.comp.decide") as decide:
            row = run_history_condition_domain(
                "division", ("addition", "multiplication"),
                "addition+multiplication", "signed", seed=0, condition_index=5,
            )

        decide.assert_not_called()
        self.assertFalse(row["prior_history_valid"])
        self.assertEqual(row["strategy"], "prerequisite_failed")
        self.assertFalse(row["target_attempted"])
        self.assertFalse(row["multiplication_prior_success"])

    def test_failed_forced_clone_source_invalidates_pair(self):
        from signed_domain_transfer import run_pair_domain, MAX_EPOCHS

        with patch("signed_domain_transfer.train_to_accuracy", return_value=(MAX_EPOCHS, False)), \
             patch("signed_domain_transfer.comp.compatibility_score") as score, \
             patch("signed_domain_transfer.comp.solve_probe_accuracy") as solve:
            row = run_pair_domain(
                "multiplication", "powers", "moderate/strong", "signed", seed=0,
            )

        score.assert_not_called()
        solve.assert_not_called()
        self.assertFalse(row["parent_reached"])
        self.assertFalse(row["pair_valid"])
        self.assertTrue(np.isnan(row["speedup"]))
        self.assertTrue(np.isnan(row["clone_epochs"]))
        self.assertTrue(np.isnan(row["relatedness_score"]))


if __name__ == "__main__":
    unittest.main()
