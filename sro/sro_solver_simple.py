"""Simplified and Fixed Implementation of the Iterative SRO algorithm."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np
from numpy.random import default_rng

from .regularizers import BaseRegularizer, NoRegularizer
from .sketching_simple import SketchConfig


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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IterativeSRO":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n_samples, n_features = X.shape

        # Initialize coefficients  
        beta = np.zeros(n_features, dtype=np.float64)
        history: List[dict[str, float]] = []

        # Precompute X^T X for efficiency (when no sketching or for small problems)
        XtX = X.T @ X / n_samples
        Xty = X.T @ y / n_samples
        
        # Compute Lipschitz constant (largest eigenvalue of X^T X)
        lipschitz = float(np.linalg.norm(XtX, ord=2))
        if lipschitz < 1e-12:
            lipschitz = 1e-12
        step_size = self.step_scale / lipschitz

        for outer_idx in range(self.max_iter):
            beta_prev = beta.copy()
            
            # Apply sketching approximation if specified
            if self.sketch_config.method != "none":
                # For sketching, we'll use a simpler approximation
                # Sample a subset of data points for gradient computation
                sketch_size = min(self.sketch_config.sketch_size or 128, n_samples)
                if sketch_size < n_samples:
                    rng = default_rng(self.random_state if not self.resample_sketch 
                                    else self._rng.integers(0, np.iinfo(np.int32).max))
                    indices = rng.choice(n_samples, size=sketch_size, replace=False)
                    X_sketch = X[indices]
                    y_sketch = y[indices]
                    
                    # Recompute approximations for this iteration
                    XtX_approx = X_sketch.T @ X_sketch / sketch_size
                    Xty_approx = X_sketch.T @ y_sketch / sketch_size
                    lipschitz_approx = float(np.linalg.norm(XtX_approx, ord=2))
                    if lipschitz_approx < 1e-12:
                        lipschitz_approx = 1e-12
                    step_size = self.step_scale / lipschitz_approx
                    
                    XtX_work = XtX_approx
                    Xty_work = Xty_approx
                else:
                    XtX_work = XtX
                    Xty_work = Xty
            else:
                XtX_work = XtX
                Xty_work = Xty
            
            # Inner loop: proximal gradient descent
            for inner_iter in range(self.inner_max_iter):
                # Compute gradient: X^T X beta - X^T y
                grad = XtX_work @ beta - Xty_work
                
                # Add numerical stability check
                if not np.isfinite(grad).all():
                    print(f"Warning: Non-finite gradient detected at outer iter {outer_idx}, inner iter {inner_iter}")
                    break
                
                # Proximal gradient step
                beta_new = self.regularizer.prox(beta - step_size * grad, step_size)
                
                # Ensure beta_new is finite
                if not np.isfinite(beta_new).all():
                    print(f"Warning: Non-finite beta detected at outer iter {outer_idx}, inner iter {inner_iter}")
                    break
                
                # Check convergence of inner loop
                change = np.linalg.norm(beta_new - beta)
                if change <= self.tol:
                    beta = beta_new
                    break
                beta = beta_new
            
            # Check convergence of outer loop
            beta_change = np.linalg.norm(beta - beta_prev)
            obj_val = self._objective(X, y, beta)
            
            history.append({
                "iteration": outer_idx + 1,
                "objective": obj_val,
                "beta_change": beta_change,
            })
            
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

    def _objective(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        residual = y - X @ beta
        loss = 0.5 * float(residual @ residual) / X.shape[0]
        penalty = self.regularizer.penalty(beta) if self.regularizer else 0.0
        return loss + penalty


__all__ = ["IterativeSRO"]
