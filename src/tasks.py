"""
tasks.py

Section 8's minimal curriculum: addition -> subtraction -> multiplication -> powers.
Each task T is a distribution over (a, b) -> op(a, b).

Design choice (not specified in the issue, chosen here):
  - operands a, b are drawn from small integer ranges so a tiny MLP can fit them
  - inputs are fed to the network scaled by /10 (keeps tanh layer out of saturation);
    targets are left in natural units since the output layer is linear
  - "powers" uses small integer base/exponent to keep targets bounded
"""
from __future__ import annotations
import numpy as np

TASK_ORDER = ["addition", "subtraction", "multiplication", "powers"]

_RANGES = {
    "addition": ((0, 10), (0, 10)),
    "subtraction": ((0, 10), (0, 10)),
    "multiplication": ((0, 10), (0, 10)),
    "powers": ((0, 5), (0, 3)),  # base 0-4, exponent 0-2
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
