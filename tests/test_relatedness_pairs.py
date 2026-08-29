"""Structural regression tests for experiments/relatedness_pairs.py.

Deliberately does not assert the *direction* of any particular pair's speedup
(e.g. that addition->subtraction is a negative-transfer case) -- that is an
empirical research finding, not an implementation invariant, and asserting it
here would make the test suite fragile to legitimate future changes in the
task definitions or training setup. What we do assert are the structural
guarantees the rest of the analysis pipeline depends on: every pair reports a
result for every requested seed, epochs are within the declared budget, and
summarize() preserves the deliberate PAIRS ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import pandas as pd  # noqa: E402
from relatedness_pairs import MAX_EPOCHS, PAIRS, run_pair, summarize  # noqa: E402


class RelatednessPairsTests(unittest.TestCase):
    def test_run_pair_returns_one_row_per_seed(self):
        rows = run_pair("multiplication", "powers", "moderate", n_seeds=2)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["pair"], "multiplication->powers")
            self.assertGreaterEqual(row["clone_epochs"], 1)
            self.assertLessEqual(row["clone_epochs"], MAX_EPOCHS)
            self.assertGreaterEqual(row["scratch_epochs"], 1)
            self.assertLessEqual(row["scratch_epochs"], MAX_EPOCHS)
            self.assertGreater(row["speedup"], 0.0)
            self.assertGreaterEqual(row["relatedness_score"], 0.0)
            self.assertLessEqual(row["relatedness_score"], 1.0)

    def test_summarize_preserves_declared_pair_order(self):
        rows = []
        for source, target, label in PAIRS:
            rows.extend(run_pair(source, target, label, n_seeds=1))
        summary = summarize(pd.DataFrame(rows))
        expected_order = [f"{s}->{t}" for s, t, _ in PAIRS]
        self.assertEqual(list(summary["pair"]), expected_order)
        self.assertEqual(len(summary), len(PAIRS))


if __name__ == "__main__":
    unittest.main()
