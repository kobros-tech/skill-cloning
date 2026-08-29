"""Regression tests for compatibility decision safety."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compatibility import ACC_SOLVE_TARGET, TAU_SOLVE, decide  # noqa: E402
from skill import Skill  # noqa: E402


class FakeNet:
    """Minimal network whose score and calibration accuracy are controlled."""

    def __init__(self, mse: float, accuracy: float):
        self._mse = mse
        self._accuracy = accuracy

    def mse(self, X, y):
        return self._mse

    def accuracy(self, X, y, tol=0.5):
        return self._accuracy


class CompatibilityDecisionTests(unittest.TestCase):
    def test_high_score_unsolved_skill_is_not_reused(self):
        # MSE=5 gives exp(-5/60) > 0.90, but the independent solve probe says
        # the target is not actually solved. The controller must clone instead
        # of silently taking the zero-training reuse path.
        skill = Skill("parent", FakeNet(mse=5.0, accuracy=ACC_SOLVE_TARGET - 0.10), origin="scratch")
        decision = decide({"parent": skill}, "powers", base_seed=123)
        self.assertGreater(decision["score"], TAU_SOLVE)
        self.assertLess(decision["solve_accuracy"], ACC_SOLVE_TARGET)
        self.assertEqual(decision["action"], "clone")

    def test_high_score_solved_skill_can_be_reused(self):
        skill = Skill("parent", FakeNet(mse=1.0, accuracy=ACC_SOLVE_TARGET), origin="scratch")
        decision = decide({"parent": skill}, "multiplication", base_seed=123)
        self.assertGreater(decision["score"], TAU_SOLVE)
        self.assertGreaterEqual(decision["solve_accuracy"], ACC_SOLVE_TARGET)
        self.assertEqual(decision["action"], "reuse")


if __name__ == "__main__":
    unittest.main()
