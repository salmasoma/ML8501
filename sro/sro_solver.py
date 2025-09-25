"""Implementation of the Iterative SRO algorithm."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np
from numpy.random import default_rng

from .regularizers import BaseRegularizer, NoRegularizer
from .sketching import SketchConfig, apply_sketch


@dataclass
class IterativeSRO:
    """Iterative SRO solver supporting convex and non-convex penalties."""

    regularizer: BaseRegularizer | None = None
    sketch_config: SketchConfig | None = None
    max_iter: int = 10
    inner_max_iter: int = 100
    tol: float = 1e-6
    step_scale: float = 1.0
    resample_sketch: bool = True
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            msg = "max_iter must be positive."
            raise ValueError(msg)
        if self.inner_max_iter <= 0:
            msg = "inner_max_iter must be positive."
            raise ValueError(msg)
        if self.tol <= 0:
            msg = "tol must be positive."
            raise ValueError(msg)
        if self.step_scale <= 0:
            msg = "step_scale must be positive."
            raise ValueError(msg)
        if self.regularizer is None:
            self.regularizer = NoRegularizer()
        if self.sketch_config is None:
            self.sketch_config = SketchConfig(method="none")

        self._rng = default_rng(self.random_state)
        self.beta_: Optional[np.ndarray] = None
        self.history_: List[dict[str, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> IterativeSRO:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        _, n_features = X.shape

        beta = np.zeros(n_features, dtype=np.float64)
        history: List[dict[str, float]] = []
        identity = np.eye(n_features, dtype=np.float64)
        global_lipschitz = float(np.linalg.norm(X, ord=2) ** 2 + 1e-12)

        for outer_idx in range(self.max_iter):
            residual = y - X @ beta
            grad_const = -X.T @ residual

            sketch_config = self._prepare_sketch_config(outer_idx)
            sketched = apply_sketch(X, sketch_config, reuse_random_state=True)
            gram = sketched.T @ sketched
            if sketch_config.method != "none":
                gram = gram + global_lipschitz * identity
            lipschitz = float(np.linalg.norm(gram, ord=2) + 1e-12)
            step_size = self.step_scale / lipschitz

            beta_prev = beta.copy()
            for _ in range(self.inner_max_iter):
                grad = gram @ (beta - beta_prev) + grad_const
                beta_next = self.regularizer.prox(beta - step_size * grad, step_size)
                if np.linalg.norm(beta_next - beta) <= self.tol:
                    beta = beta_next
                    break
                beta = beta_next

            beta_change = np.linalg.norm(beta - beta_prev)
            obj_val = self._objective(X, y, beta)
            history.append(
                {
                    "iteration": outer_idx + 1,
                    "objective": obj_val,
                    "beta_change": beta_change,
                }
            )

            if beta_change <= self.tol:
                break

        self.beta_ = beta
        self.history_ = history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            msg = "Model has not been fitted yet."
            raise RuntimeError(msg)
        X = np.asarray(X, dtype=np.float64)
        return X @ self.beta_

    def get_history(self) -> List[dict[str, float]]:
        return list(self.history_)

    def _prepare_sketch_config(self, iteration_index: int) -> SketchConfig:
        assert self.sketch_config is not None
        if self.sketch_config.method == "none":
            return SketchConfig(method="none")

        config = replace(self.sketch_config)
        if self.resample_sketch:
            config.random_state = None if self.random_state is None else int(
                self._rng.integers(0, np.iinfo(np.int32).max)
            )
        return config

    def _objective(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        residual = y - X @ beta
        loss = 0.5 * float(residual @ residual)
        penalty = self.regularizer.penalty(beta) if self.regularizer else 0.0
        return loss + penalty


__all__ = ["IterativeSRO"]
