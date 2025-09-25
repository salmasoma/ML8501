"""Implementation of the Iterative SRO algorithm."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional

import numpy as np
from numpy.linalg import norm
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
    step_scale: float = 0.25
    resample_sketch: bool = True
    random_state: Optional[int] = None
    sketch_stabiliser: float = 1e-3

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
        if self.sketch_stabiliser < 0:
            msg = "sketch_stabiliser must be non-negative."
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
        n_samples, n_features = X.shape

        if n_samples == 0 or n_features == 0:
            msg = "X must contain at least one sample and one feature."
            raise ValueError(msg)

        scale = 1.0 / float(n_samples)

        beta = np.zeros(n_features, dtype=np.float64)
        history: List[dict[str, float]] = []

        for outer_idx in range(self.max_iter):
            residual = X @ beta - y
            grad_const = scale * (X.T @ residual)

            sketch_config = self._prepare_sketch_config(outer_idx)
            sketched = apply_sketch(X, sketch_config, reuse_random_state=True)

            hessian_mv = self._build_hessian_matvec(
                X,
                sketched,
                scale,
                sketch_config.method,
            )
            lipschitz = self._estimate_lipschitz(hessian_mv, n_features)
            step_size = self.step_scale / max(lipschitz, 1e-12)

            beta_prev = beta.copy()
            for _ in range(self.inner_max_iter):
                grad = hessian_mv(beta - beta_prev) + grad_const
                beta_next = self.regularizer.prox(beta - step_size * grad, step_size)
                if norm(beta_next - beta) <= self.tol:
                    beta = beta_next
                    break
                beta = beta_next

            beta_change = norm(beta - beta_prev)
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
        n_samples = float(X.shape[0])
        residual = X @ beta - y
        loss = 0.5 / n_samples * float(residual @ residual)
        penalty = self.regularizer.penalty(beta) if self.regularizer else 0.0
        return loss + penalty

    def _build_hessian_matvec(
        self,
        X: np.ndarray,
        sketched: np.ndarray,
        scale: float,
        method: str,
    ) -> Callable[[np.ndarray], np.ndarray]:
        stabiliser = 0.0 if method == "none" else float(self.sketch_stabiliser)

        if method == "none":

            def matvec(vec: np.ndarray) -> np.ndarray:
                return scale * (X.T @ (X @ vec))

            return matvec

        def matvec(vec: np.ndarray) -> np.ndarray:
            return scale * (sketched.T @ (sketched @ vec)) + stabiliser * vec

        return matvec

    def _estimate_lipschitz(
        self,
        matvec: Callable[[np.ndarray], np.ndarray],
        n_features: int,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> float:
        vec = self._rng.normal(size=n_features)
        vec_norm = norm(vec)
        if vec_norm == 0:
            return 1.0
        vec /= vec_norm

        eigenvalue = 0.0
        for _ in range(max_iter):
            next_vec = matvec(vec)
            next_norm = norm(next_vec)
            if next_norm == 0:
                return max(eigenvalue, 1.0)
            vec = next_vec / next_norm
            if abs(next_norm - eigenvalue) <= tol * max(1.0, eigenvalue):
                eigenvalue = next_norm
                break
            eigenvalue = next_norm

        return max(eigenvalue, 1.0)


__all__ = ["IterativeSRO"]
