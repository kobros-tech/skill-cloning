"""Regression tests for tasks.py's domain configuration and the signed-domain
transfer experiment's plumbing.

Per the plan this implements: these tests verify implementation invariants
of the domain mechanism (no negative operands leak into "nonnegative" mode,
negative operands actually appear in "signed" mode, division never samples a
zero divisor, powers never gets a negative exponent, scaling stays
consistent, sample counts stay controlled, and both domains are
deterministic/reproducible under a fixed seed). They deliberately do NOT
assert any empirical transfer direction (e.g. "multiplication->powers must
be faster than scratch") -- that is a scientific result reported in
results/signed_domain_pairs_summary.csv and docs/final_paper.md, not an
implementation invariant, and pinning it here would make the suite fragile
to legitimate future changes in the research findings themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

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
            self.assertTrue((X >= 0).all(), f"{task} produced a negative operand under nonnegative domain")

    def test_signed_domain_produces_negative_operands(self):
        for task in ["addition", "subtraction", "multiplication", "squares", "division"]:
            X, _ = sample_task(task, 500, seed=1, domain="signed")
            self.assertTrue((X < 0).any(), f"{task} never produced a negative operand under signed domain")

    def test_signed_powers_base_can_be_negative_but_exponent_cannot(self):
        X, _ = sample_task("powers", 2000, seed=1, domain="signed")
        base_scaled, exponent_scaled = X[:, 0], X[:, 1]
        self.assertTrue((base_scaled < 0).any(), "signed powers base was never negative")
        self.assertTrue((exponent_scaled >= 0).all(), "signed powers exponent went negative")

    def test_division_never_samples_a_zero_divisor(self):
        for domain in DOMAINS:
            X, y = sample_task("division", 2000, seed=1, domain=domain)
            divisor_scaled = X[:, 1]
            self.assertTrue((divisor_scaled != 0).all(), f"{domain} division sampled a zero divisor")
            self.assertTrue(np.isfinite(y).all(), f"{domain} division produced a non-finite target")

    def test_signed_division_divisor_covers_both_signs(self):
        X, _ = sample_task("division", 2000, seed=1, domain="signed")
        divisor_scaled = X[:, 1]
        self.assertTrue((divisor_scaled > 0).any())
        self.assertTrue((divisor_scaled < 0).any())

    def test_input_scaling_is_consistent_across_domains(self):
        # Every task uses the same /10 scaling in both domains -- spot-check
        # that raw operands recovered from X * 10 are integers.
        for domain in DOMAINS:
            for task in ["addition", "multiplication", "squares"]:
                X, _ = sample_task(task, 200, seed=3, domain=domain)
                raw = X * 10.0
                self.assertTrue(np.allclose(raw, np.round(raw)), f"{task}/{domain} scaling inconsistent")

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

    def test_nonnegative_domain_matches_original_task_ranges_exactly(self):
        """The nonnegative domain must reproduce the exact pre-existing
        random-draw sequence, since every prior experiment's results (PR #1
        through PR #6) depend on this being unchanged."""
        X, y = sample_task("division", 10, seed=42, domain="nonnegative")
        # Recompute independently using the historical range definition.
        rng = np.random.default_rng(42)
        a = rng.integers(0, 25, size=10)
        b = rng.integers(1, 6, size=10)
        expected_y = a.astype(float) / b.astype(float)
        expected_X = np.stack([a, b], axis=1).astype(float) / 10.0
        self.assertTrue(np.array_equal(X, expected_X))
        self.assertTrue(np.array_equal(y, expected_y))


class CompatibilityDomainThreadingTests(unittest.TestCase):
    """The compatibility probes must actually use the requested domain --
    this is the bug caught during implementation (probes were hardcoded to
    "nonnegative" before domain= was threaded through)."""

    def test_compatibility_score_uses_the_requested_domain(self):
        import compatibility as comp
        from skill import TinyMLP

        net = TinyMLP(hidden_dim=8, seed=1)
        score_nonneg = comp.compatibility_score(net, "multiplication", base_seed=1, domain="nonnegative")
        score_signed = comp.compatibility_score(net, "multiplication", base_seed=1, domain="signed")
        # An untrained random network scored against two different data
        # distributions should not coincidentally produce the exact same
        # float -- if it does, the domain argument is not reaching the probe.
        self.assertNotEqual(score_nonneg, score_signed)

    def test_decide_defaults_to_nonnegative_domain(self):
        """decide() with no domain argument must behave exactly as it did
        before the domain mechanism was added (every existing caller relies
        on this)."""
        import inspect
        import compatibility as comp
        sig = inspect.signature(comp.decide)
        self.assertEqual(sig.parameters["domain"].default, "nonnegative")


if __name__ == "__main__":
    unittest.main()
