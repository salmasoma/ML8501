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
    if method not in {"none", "gaussian", "count", "count_gaussian", "srht"}:
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

        if method == "count_gaussian":
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

        # method == "srht"
        m = int(config.sketch_size)
        if m <= 0 or m > n_samples:
            msg = "SRHT sketch size must satisfy 1 <= m <= n_samples."
            raise ValueError(msg)
        return _apply_srht(X_array, m, rng)
    finally:
        if reuse_random_state and config.random_state is not None:
            np.random.set_state(state)


def _ensure_c_contiguous(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(X):
        X = X.toarray()
    return np.ascontiguousarray(X, dtype=np.float64)


def _apply_srht(X: np.ndarray, sketch_size: int, rng: Generator) -> np.ndarray:
    """Apply a subsampled randomized Hadamard transform to ``X``."""
    n_samples, _ = X.shape
    padded = 1 << (n_samples - 1).bit_length()
    if padded != n_samples:
        pad = padded - n_samples
        X_work = np.pad(X, ((0, pad), (0, 0)), mode="constant")
    else:
        X_work = X.copy()

    signs = rng.choice((-1.0, 1.0), size=padded)
    X_work *= signs[:, None]

    transformed = _fast_hadamard_transform(X_work) / np.sqrt(padded)
    transformed = transformed[:n_samples]
    indices = rng.choice(n_samples, size=sketch_size, replace=False)
    scaling = np.sqrt(n_samples / sketch_size)
    return transformed[indices] * scaling


def _fast_hadamard_transform(X: np.ndarray) -> np.ndarray:
    """Compute the Walsh-Hadamard transform along the first axis."""
    H = X.copy()
    n_rows = H.shape[0]
    if n_rows & (n_rows - 1) != 0:
        msg = "Fast Hadamard transform requires the number of rows to be a power of two."
        raise ValueError(msg)

    step = 1
    while step < n_rows:
        span = step * 2
        for start in range(0, n_rows, span):
            mid = start + step
            end = start + span
            temp = H[start:mid] + H[mid:end]
            H[mid:end] = H[start:mid] - H[mid:end]
            H[start:mid] = temp
        step = span
    return H


__all__ = ["SketchConfig", "apply_sketch"]
