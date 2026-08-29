"""
skill.py

Minimal parametric "skill" as specified in Phase 1/2 of the design doc:
  - skill representation:      a tiny 2-layer MLP, W_i in R^{d_i}
  - loss function:             mean squared error (regression on scalar outputs)
  - parameter update rule:     Adam gradient descent
  - cloning operation:         deep-copy of parameters (Section 5)

Deliberately dependency-free (numpy only) and small, per the issue's
"no unnecessary abstractions" instruction.
"""
from __future__ import annotations
import numpy as np
import copy


class TinyMLP:
    """A 2-2-32-1 style regression network: inputs -> hidden(tanh) -> scalar output."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, seed: int | None = None):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.params = {
            "W1": rng.normal(0, scale1, size=(hidden_dim, input_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(0, scale2, size=(1, hidden_dim)),
            "b2": np.zeros(1),
        }
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._adam_state = self._init_adam_state()

    # ---- optimizer state -------------------------------------------------
    def _init_adam_state(self):
        return {
            k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
            for k, v in self.params.items()
        }

    def reset_optimizer(self):
        """Reset Adam moment estimates (used when a skill starts training on a new task)."""
        self._adam_state = self._init_adam_state()

    # ---- forward / backward -----------------------------------------------
    def forward(self, X: np.ndarray):
        """X: (N, input_dim). Returns (pred, cache) with pred shape (N,)."""
        z1 = X @ self.params["W1"].T + self.params["b1"]      # (N, H)
        h1 = np.tanh(z1)
        z2 = h1 @ self.params["W2"].T + self.params["b2"]     # (N, 1)
        pred = z2.ravel()
        cache = (X, z1, h1)
        return pred, cache

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred, _ = self.forward(X)
        return pred

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean((pred - y) ** 2))

    def accuracy(self, X: np.ndarray, y: np.ndarray, tol: float = 0.5) -> float:
        pred = self.predict(X)
        return float(np.mean(np.abs(pred - y) <= tol))

    def _backward(self, X, y, cache):
        N = X.shape[0]
        _, z1, h1 = cache
        pred, _ = self.forward(X)
        dpred = (2.0 / N) * (pred - y)               # (N,)
        dz2 = dpred.reshape(-1, 1)                    # (N, 1)
        dW2 = dz2.T @ h1                               # (1, H)
        db2 = dz2.sum(axis=0)                          # (1,)
        dh1 = dz2 @ self.params["W2"]                  # (N, H)
        dz1 = dh1 * (1 - h1 ** 2)                       # tanh'
        dW1 = dz1.T @ X                                 # (H, input_dim)
        db1 = dz1.sum(axis=0)                           # (H,)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---- training step (Adam), with an optional "trainable" mask ----------
    def train_step(self, X, y, lr=0.01, trainable=None, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        One full-batch Adam step. `trainable`: iterable of param names allowed to
        update (e.g. only the clone's params, never the frozen parent's).
        """
        grads = self._backward(X, y, self.forward(X)[1])
        trainable = set(self.params.keys()) if trainable is None else set(trainable)
        for k in self.params:
            if k not in trainable:
                continue
            st = self._adam_state[k]
            st["t"] += 1
            st["m"] = beta1 * st["m"] + (1 - beta1) * grads[k]
            st["v"] = beta2 * st["v"] + (1 - beta2) * (grads[k] ** 2)
            m_hat = st["m"] / (1 - beta1 ** st["t"])
            v_hat = st["v"] / (1 - beta2 ** st["t"])
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def clone(self) -> "TinyMLP":
        """Section 5: W_new^(0) = W_j  (deep copy, fresh optimizer state)."""
        new = TinyMLP.__new__(TinyMLP)
        new.params = copy.deepcopy(self.params)
        new.input_dim = self.input_dim
        new.hidden_dim = self.hidden_dim
        new._adam_state = new._init_adam_state()
        return new

    def num_params(self) -> int:
        return sum(v.size for v in self.params.values())


class Skill:
    """A named skill = a TinyMLP + bookkeeping (which task it was created for, lineage)."""

    def __init__(self, name: str, net: TinyMLP, origin: str = "scratch", parent: str | None = None):
        self.name = name
        self.net = net
        self.origin = origin      # "scratch" | "clone" | "reuse"
        self.parent = parent      # name of parent skill if cloned
        self.tasks_covered = [name]  # tasks this skill is responsible for (grows on "reuse")

    def train(self, X, y, epochs, lr, trainable=None, target_mse=None, log_every=1):
        """Train for up to `epochs` full-batch Adam steps. Returns loss history."""
        history = []
        for e in range(epochs):
            self.net.train_step(X, y, lr=lr, trainable=trainable)
            if e % log_every == 0 or e == epochs - 1:
                loss = self.net.mse(X, y)
                history.append(loss)
                if target_mse is not None and loss <= target_mse:
                    break
        return history
