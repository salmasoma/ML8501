"""Simple sketching utilities without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator, default_rng


@dataclass
class SketchConfig:
    """Configuration for generating a sketch matrix."""

    method: str = "none"
    sketch_size: Optional[int] = None
    count_size: Optional[int] = None
    random_state: Optional[int] = None


def apply_sketch(
    X: np.ndarray,
    config: SketchConfig,
    *,
    reuse_random_state: bool = False,
) -> np.ndarray:
    """Apply a sketch to ``X`` according to ``config`` and return ``S @ X``.

    Parameters
    ----------
    X:
        Input matrix with shape ``(n_samples, n_features)``.
    config:
        Sketch configuration describing which transform to apply.
    reuse_random_state:
        When ``True`` the global NumPy RNG is restored after applying the
        sketch. This is useful when deterministic behaviour is required.
    """

    method = config.method.lower()
    if method not in {"none", "gaussian", "subsampling"}:
        msg = f"Unknown sketching method '{config.method}'. Available: none, gaussian, subsampling"
        raise ValueError(msg)

    if method == "none" or config.sketch_size in {None, 0}:
        return np.asarray(X, dtype=np.float64)

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, _ = X_array.shape

    rng: Generator = default_rng(config.random_state)
    if reuse_random_state and config.random_state is not None:
        state = np.random.get_state()
        np.random.seed(config.random_state)

    try:
        if method == "gaussian":
            m = int(config.sketch_size)
            if m <= 0 or m > n_samples:
                msg = (
                    "Gaussian sketch size must be in the range ``1..n_samples``."
                )
                raise ValueError(msg)
            projection = rng.normal(size=(m, n_samples)) / np.sqrt(m)
            return projection @ X_array

        elif method == "subsampling":
            # Simple uniform subsampling
            m = int(config.sketch_size)
            if m <= 0 or m > n_samples:
                msg = "Subsampling size must satisfy 1 <= m <= n_samples."
                raise ValueError(msg)
            indices = rng.choice(n_samples, size=m, replace=False)
            return X_array[indices] * np.sqrt(n_samples / m)  # Rescale for unbiasedness
        
    finally:
        if reuse_random_state and config.random_state is not None:
            np.random.set_state(state)

    # Should never reach here
    return X_array


__all__ = ["SketchConfig", "apply_sketch"]
