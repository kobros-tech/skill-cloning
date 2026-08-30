"""
tasks.py

Section 8's minimal curriculum: addition -> subtraction -> multiplication -> powers.
Each task T is a distribution over (a, b) -> op(a, b).

The prerequisite-acquisition follow-up also uses division and squares as target
experiments without changing TASK_ORDER, so the historical Phase 4 curriculum
results remain reproducible.

Domain configuration (signed-domain follow-up):
  sample_task() takes an explicit `domain` argument, defaulting to
  "nonnegative" -- the domain every existing experiment and every prior
  result in this repository was generated under. That default preserves the
  exact original random-number-generator call sequence for every task, so
  every existing script's numbers remain bit-for-bit reproducible without
  passing anything new. A second domain, "signed", extends operand ranges to
  include negative values (see DOMAIN_RANGES / DIVISION_RANGES below) for
  experiments/signed_domain_transfer.py. Nothing about the default path
  changes: this is additive, not a redefinition of the existing behavior.

Design choices:
  - operands are drawn from small integer ranges so a tiny MLP can fit them
  - inputs are scaled by /10 to keep the tanh layer out of saturation
  - targets are left in natural units since the output layer is linear
  - powers uses small base/exponent values to keep targets bounded; the
    exponent is kept non-negative in both domains (a negative exponent would
    introduce fractional targets, changing the learning problem for reasons
    unrelated to operand sign -- see experiments/signed_domain_transfer.py)
"""
from __future__ import annotations
import numpy as np

TASK_ORDER = ["addition", "subtraction", "multiplication", "powers"]

DOMAINS = ("nonnegative", "signed")

# Per-domain (a_range, b_range) for every task except division, which needs a
# nonzero-divisor constraint handled separately (see DIVISION_RANGES).
DOMAIN_RANGES = {
    "nonnegative": {
        "addition": ((0, 10), (0, 10)),
        "subtraction": ((0, 10), (0, 10)),
        "multiplication": ((0, 10), (0, 10)),
        "powers": ((0, 5), (0, 3)),  # base 0-4, exponent 0-2
        "squares": ((0, 10), (0, 10)),
    },
    "signed": {
        "addition": ((-9, 10), (-9, 10)),
        "subtraction": ((-9, 10), (-9, 10)),
        "multiplication": ((-9, 10), (-9, 10)),
        "powers": ((-4, 5), (0, 3)),  # base -4..4; exponent stays 0-2 (see module docstring)
        "squares": ((-9, 10), (-9, 10)),
    },
}

# Division is special-cased in both domains: the divisor must never be zero.
# nonnegative: divisor drawn from {1,...,5}. signed: divisor's magnitude is
# drawn from {1,...,5} and its sign independently, so {-5,...,-1} u {1,...,5}
# with none of the mass sitting at zero.
DIVISION_RANGES = {
    "nonnegative": {"numerator": (0, 25), "divisor": (1, 6)},
    "signed": {"numerator": (-24, 25), "divisor_magnitude": (1, 6)},
}

# Kept for anything that still imports the old private name directly.
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
        # A deliberately simple intermediate task: square the first operand.
        # It is structurally related to multiplication while leaving the second
        # input available so all tasks share the same model interface.
        return np.square(a.astype(float))
    if task == "division":
        # Divisor is guaranteed nonzero by construction in both domains.
        # Fractional outputs are intentional: division should be a distinct
        # target, not integer quotient classification disguised as arithmetic.
        return a.astype(float) / b.astype(float)
    raise ValueError(f"unknown task {task}")


def sample_task(task: str, n: int, seed: int, domain: str = "nonnegative") -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X shape (n, 2) scaled for network input, y shape (n,) raw targets.

    `domain="nonnegative"` (the default) reproduces the exact original
    random-draw sequence for every task -- every experiment result generated
    before the signed-domain follow-up remains reproducible without any
    caller needing to pass this argument.
    """
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
