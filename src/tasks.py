"""
tasks.py

Section 8's minimal curriculum: addition -> subtraction -> multiplication -> powers.
Each task T is a distribution over (a, b) -> op(a, b).

The prerequisite-acquisition follow-up also uses division and squares as target
experiments without changing TASK_ORDER, so the historical Phase 4 curriculum
results remain reproducible.

Design choices:
  - operands are drawn from small integer ranges so a tiny MLP can fit them
  - inputs are scaled by /10 to keep the tanh layer out of saturation
  - targets are left in natural units since the output layer is linear
  - powers uses small base/exponent values to keep targets bounded
"""
from __future__ import annotations
import numpy as np

TASK_ORDER = ["addition", "subtraction", "multiplication", "powers"]

_RANGES = {
    "addition": ((0, 10), (0, 10)),
    "subtraction": ((0, 10), (0, 10)),
    "multiplication": ((0, 10), (0, 10)),
    "powers": ((0, 5), (0, 3)),  # base 0-4, exponent 0-2
    "squares": ((0, 10), (0, 10)),
    "division": ((0, 25), (1, 6)),  # nonzero divisor, bounded quotient
}


def _op(task: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if task == "addition":
        return a + b
    if task == "subtraction":
        return a - b
    if task == "multiplication":
        return a * b
    if task == "powers":
        return np.power(a.astype(float), b.astype(float))
    if task == "squares":
        # A deliberately simple intermediate task: square the first operand.
        # It is structurally related to multiplication while leaving the second
        # input available so all tasks share the same model interface.
        return np.square(a.astype(float))
    if task == "division":
        # Divisor is guaranteed nonzero by _RANGES.  Fractional outputs are
        # intentional: division should be a distinct target, not integer
        # quotient classification disguised as arithmetic.
        return a.astype(float) / b.astype(float)
    raise ValueError(f"unknown task {task}")


def sample_task(task: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X shape (n, 2) scaled for network input, y shape (n,) raw targets."""
    rng = np.random.default_rng(seed)
    (a_lo, a_hi), (b_lo, b_hi) = _RANGES[task]
    a = rng.integers(a_lo, a_hi, size=n)
    b = rng.integers(b_lo, b_hi, size=n)
    y = _op(task, a, b).astype(float)
    X = np.stack([a, b], axis=1).astype(float) / 10.0
    return X, y
