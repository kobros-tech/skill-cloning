"""Structural tests for the skill-isolation invariant check (experiments/retention.py).

Reframed after review: this file's retention_delta is zero by construction
under the current architecture (frozen skills are never touched again), so
these tests check the invariant itself and the plumbing around it -- they
are not testing a statistical experiment, and none of them assert a
bootstrap CI or effect size (that machinery was removed from retention.py
for the same reason: it has nothing meaningful to describe here).
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import pandas as pd  # noqa: E402
from retention import (  # noqa: E402
    MAX_EPOCHS,
    RETENTION_TOLERANCE,
    SEQUENCES,
    run_sequence,
    summarize,
)


class RetentionTests(unittest.TestCase):
    def test_sequences_use_existing_task_family(self):
        self.assertEqual(len(SEQUENCES), 4)
        for sequence in SEQUENCES:
            self.assertEqual(len(sequence), 3)
            self.assertEqual(sequence[0], "addition")

    def test_retention_evaluation_is_paired_on_a_stable_set(self):
        rows = run_sequence(SEQUENCES[0], seed=0, sequence_index=0)
        frame = pd.DataFrame(rows)
        self.assertTrue(frame["is_retention_check"].any())
        for skill in frame["evaluated_skill"].unique():
            skill_rows = frame[frame["evaluated_skill"] == skill]
            if len(skill_rows) > 1:
                self.assertEqual(
                    skill_rows["pre_accuracy"].iloc[0],
                    skill_rows["pre_accuracy"].iloc[-1],
                )
                self.assertEqual(
                    skill_rows["post_accuracy"].iloc[0],
                    skill_rows["post_accuracy"].iloc[-1],
                )

    def test_retention_summary_excludes_baseline_rows(self):
        rows = run_sequence(SEQUENCES[0], seed=0, sequence_index=0)
        raw = pd.DataFrame(rows)
        summary = summarize(raw)
        self.assertFalse(summary.empty)
        self.assertTrue((summary["stage"] > summary["evaluated_skill"].map(
            {"addition": 0, "multiplication": 1, "division": 2, "subtraction": 1, "squares": 2, "powers": 2}
        )).all())
        self.assertTrue((summary["retention_tolerance"] == RETENTION_TOLERANCE).all())
        self.assertTrue((summary["n_runs"] >= 1).all())

    def test_retention_delta_and_pass_are_consistent(self):
        rows = run_sequence(SEQUENCES[0], seed=0, sequence_index=0)
        frame = pd.DataFrame(rows)
        checks = frame[frame["is_retention_check"]]
        for row in checks.itertuples():
            self.assertAlmostEqual(row.retention_delta, row.post_accuracy - row.pre_accuracy)
            self.assertEqual(
                row.retention_pass,
                row.retention_delta >= -RETENTION_TOLERANCE,
            )
        self.assertLessEqual(checks["adaptation_steps"].max(), MAX_EPOCHS)

    def test_isolation_invariant_holds_exactly(self):
        """This is the invariant this file actually exists to check: a frozen
        skill's accuracy on its own stable eval set must be bit-identical
        before and after any later acquisition. A failure here would mean a
        real bug -- some later training step leaking into a skill that is
        supposed to be frozen -- not sampling noise, since there is no
        randomness in re-evaluating a fixed network on a fixed batch."""
        rows = run_sequence(SEQUENCES[0], seed=0, sequence_index=0)
        frame = pd.DataFrame(rows)
        checks = frame[frame["is_retention_check"]]
        self.assertTrue((checks["retention_delta"] == 0.0).all())

    def test_summary_reports_invariant_not_statistics(self):
        """The summary should describe the invariant directly (whether every
        check landed at delta 0) rather than statistical uncertainty over a
        quantity that has none -- so bootstrap/effect-size columns must not
        reappear here."""
        rows = run_sequence(SEQUENCES[0], seed=0, sequence_index=0)
        summary = summarize(pd.DataFrame(rows))
        self.assertIn("all_deltas_exactly_zero", summary.columns)
        self.assertIn("max_absolute_retention_delta", summary.columns)
        self.assertNotIn("bootstrap_ci_low", summary.columns)
        self.assertNotIn("bootstrap_ci_high", summary.columns)
        self.assertNotIn("paired_effect_size", summary.columns)
        self.assertTrue(summary["all_deltas_exactly_zero"].all())


if __name__ == "__main__":
    unittest.main()
