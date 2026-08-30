"""
tasks.py

Section 8's minimal curriculum: addition -> subtraction -> multiplication -> powers.
Each task T is a distribution over (a, b) -> op(a, b).

The prerequisite-acquisition follow-up also uses division and squares as target
experiments without changing TASK_ORDER, so the historical Phase 4 curriculum
results remain reproducible.

Domain configuration (signed-domain follow-up):
  sample_task() takes an explicit `domain` argument, defaulting to
  "nonnegative". That default preserves the original random-number generator
  call sequence and therefore the existing non-negative experiments.

Design choices:
  - operands are drawn from small integer ranges so a tiny MLP can fit them
  - inputs are scaled by /10 to keep the tanh layer out of saturation
  - targets are left in natural units since the output layer is linear
  - powers uses small base/exponent values; the exponent stays non-negative
    in both domains so the signed comparison changes operand sign without
    introducing fractional targets
"""
from __future__ import annotations
import numpy as np

TASK_ORDER = ["addition", "subtraction", "multiplication", "powers"]

DOMAINS = ("nonnegative", "signed")

DOMAIN_RANGES = {
    "nonnegative": {
        "addition": ((0, 10), (0, 10)),
        "subtraction": ((0, 10), (0, 10)),
        "multiplication": ((0, 10), (0, 10)),
        "powers": ((0, 5), (0, 3)),
        "squares": ((0, 10), (0, 10)),
    },
    "signed": {
        "addition": ((-9, 10), (-9, 10)),
        "subtraction": ((-9, 10), (-9, 10)),
        "multiplication": ((-9, 10), (-9, 10)),
        "powers": ((-4, 5), (0, 3)),
        "squares": ((-9, 10), (-9, 10)),
    },
}

DIVISION_RANGES = {
    "nonnegative": {"numerator": (0, 25), "divisor": (1, 6)},
    "signed": {"numerator": (-24, 25), "divisor_magnitude": (1, 6)},
}

# Kept for compatibility with any existing code importing the historical name.
_RANGES = {**DOMAIN_RANGES["nonnegative"], "division": ((0, 25), (1, 6))}


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
        return np.square(a.astype(float))
    if task == "division":
        return a.astype(float) / b.astype(float)
    raise ValueError(f"unknown task {task}")


def sample_task(
    task: str,
    n: int,
    seed: int,
    domain: str = "nonnegative",
) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled inputs and raw targets for the requested task/domain."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")

    rng = np.random.default_rng(seed)

    if task == "division":
        if domain == "nonnegative":
            a_lo, a_hi = DIVISION_RANGES["nonnegative"]["numerator"]
            b_lo, b_hi = DIVISION_RANGES["nonnegative"]["divisor"]
            a = rng.integers(a_lo, a_hi, size=n)
            b = rng.integers(b_lo, b_hi, size=n)
        else:
            a_lo, a_hi = DIVISION_RANGES["signed"]["numerator"]
            m_lo, m_hi = DIVISION_RANGES["signed"]["divisor_magnitude"]
            a = rng.integers(a_lo, a_hi, size=n)
            magnitude = rng.integers(m_lo, m_hi, size=n)
            sign = rng.choice(np.array([-1, 1]), size=n)
            b = magnitude * sign
        y = _op(task, a, b).astype(float)
        X = np.stack([a, b], axis=1).astype(float) / 10.0
        return X, y

    (a_lo, a_hi), (b_lo, b_hi) = DOMAIN_RANGES[domain][task]
    a = rng.integers(a_lo, a_hi, size=n)
    b = rng.integers(b_lo, b_hi, size=n)
    y = _op(task, a, b).astype(float)
    X = np.stack([a, b], axis=1).astype(float) / 10.0
    return X, y
