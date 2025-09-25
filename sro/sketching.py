"""Sketching utilities leveraging :mod:`pylspack`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator, default_rng
from scipy import sparse

try:
    from pylspack import leverage_scores
except ImportError as exc:  # pragma: no cover - handled at runtime.
    raise ImportError(
        "pylspack must be installed to use sketching utilities."
    ) from exc


@dataclass
class SketchConfig:
    """Configuration for generating a sketch matrix."""

    method: str = "none"
    sketch_size: Optional[int] = None
    count_size: Optional[int] = None
    random_state: Optional[int] = None


def apply_sketch(
    X: np.ndarray | sparse.spmatrix,
    config: SketchConfig,
    *,
    reuse_random_state: bool = False,
) -> np.ndarray:
    """Apply a sketch to ``X`` according to ``config`` and return ``S @ X``.

    Parameters
    ----------
    X:
        Input matrix with shape ``(n_samples, n_features)``. Dense and sparse
        inputs are supported and converted to ``float64``.
    config:
        Sketch configuration describing which transform to apply.
    reuse_random_state:
        When ``True`` the global NumPy RNG is restored after applying the
        sketch. This is useful when deterministic behaviour is required.
    """

    method = config.method.lower()
    if method not in {"none", "gaussian", "count", "count_gaussian"}:
        msg = f"Unknown sketching method '{config.method}'."
        raise ValueError(msg)

    if method == "none" or config.sketch_size in {None, 0}:
        return np.asarray(X, dtype=np.float64)

    X_array = _ensure_c_contiguous(X)
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

        if method == "count":
            r = int(config.sketch_size)
            if r <= 0 or r > n_samples:
                msg = "Count sketch size must satisfy 1 <= r <= n_samples."
                raise ValueError(msg)
            X_csr = sparse.csr_matrix(X_array)
            return leverage_scores.csrcgs(X_csr, m=0, r=r)

        # method == "count_gaussian"
        m = int(config.sketch_size)
        r = config.count_size if config.count_size is not None else min(n_samples, 2 * m)
        if m <= 0 or m > n_samples:
            msg = "Gaussian rows must satisfy 1 <= m <= n_samples."
            raise ValueError(msg)
        if r <= 0 or r > n_samples:
            msg = "Count sketch rows must satisfy 1 <= r <= n_samples."
            raise ValueError(msg)
        X_csr = sparse.csr_matrix(X_array)
        return leverage_scores.csrcgs(X_csr, m=m, r=r)
    finally:
        if reuse_random_state and config.random_state is not None:
            np.random.set_state(state)


def _ensure_c_contiguous(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(X):
        X = X.toarray()
    return np.ascontiguousarray(X, dtype=np.float64)


__all__ = ["SketchConfig", "apply_sketch"]
