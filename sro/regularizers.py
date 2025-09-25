"""Regularizer implementations used by the iterative SRO solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class BaseRegularizer(Protocol):
    """Interface for penalties that can be used with :class:`IterativeSRO`."""

    strength: float

    def prox(self, z: np.ndarray, step_size: float) -> np.ndarray:
        """Return the proximal update ``argmin_x 0.5||x - z||^2 + step_size * h(x)``."""

    def penalty(self, beta: np.ndarray) -> float:
        """Evaluate ``h(beta)`` for reporting metrics."""


@dataclass(frozen=True)
class NoRegularizer:
    """No regularisation."""

    strength: float = 0.0

    def prox(self, z: np.ndarray, step_size: float) -> np.ndarray:  # noqa: D401
        return np.asarray(z, dtype=float)

    def penalty(self, beta: np.ndarray) -> float:  # noqa: D401
        return 0.0


@dataclass(frozen=True)
class L2Regularizer:
    """Quadratic (ridge) penalty ``lambda / 2 * ||beta||_2^2``."""

    strength: float

    def prox(self, z: np.ndarray, step_size: float) -> np.ndarray:  # noqa: D401
        denom = 1.0 + step_size * self.strength
        return np.asarray(z, dtype=float) / denom

    def penalty(self, beta: np.ndarray) -> float:  # noqa: D401
        beta = np.asarray(beta, dtype=float)
        return 0.5 * self.strength * float(beta @ beta)


@dataclass(frozen=True)
class L1Regularizer:
    """Lasso penalty ``lambda * ||beta||_1``."""

    strength: float

    def prox(self, z: np.ndarray, step_size: float) -> np.ndarray:  # noqa: D401
        z = np.asarray(z, dtype=float)
        thresh = self.strength * step_size
        return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)

    def penalty(self, beta: np.ndarray) -> float:  # noqa: D401
        return self.strength * float(np.linalg.norm(beta, ord=1))


@dataclass(frozen=True)
class SCADRegularizer:
    """Smoothly clipped absolute deviation (SCAD) penalty."""

    strength: float
    a: float = 3.7

    def __post_init__(self) -> None:
        if self.a <= 2.0:
            msg = "The SCAD parameter 'a' must be greater than 2."
            raise ValueError(msg)

    def prox(self, z: np.ndarray, step_size: float) -> np.ndarray:  # noqa: D401
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        for idx, value in np.ndenumerate(z):
            out[idx] = _scad_prox_scalar(float(value), step_size, self.strength, self.a)
        return out

    def penalty(self, beta: np.ndarray) -> float:  # noqa: D401
        beta = np.asarray(beta, dtype=float)
        lam = self.strength
        a = self.a
        abs_beta = np.abs(beta)
        penalties = np.where(
            abs_beta <= lam,
            lam * abs_beta,
            np.where(
                abs_beta <= a * lam,
                (-(abs_beta**2) + 2 * a * lam * abs_beta - lam**2) / (2 * (a - 1)),
                0.5 * (a + 1) * lam**2,
            ),
        )
        return float(np.sum(penalties))


def _scad_prox_scalar(z: float, step_size: float, lam: float, a: float) -> float:
    """Compute the proximal operator of the SCAD penalty for a single value."""

    if lam <= 0:
        return z

    candidates = set()
    candidates.add(0.0)

    sign = 1.0 if z >= 0 else -1.0
    abs_z = abs(z)
    thresh = lam * step_size

    # Region 1: behaves like soft-thresholding.
    soft = sign * max(abs_z - thresh, 0.0)
    if abs(soft) <= lam + 1e-12:
        candidates.add(soft)

    # Region 2: analytical solution when lambda < |beta| <= a * lambda.
    denom = (a - 1) - step_size
    if denom > 1e-12:
        numerator = (a - 1) * abs_z - step_size * a * lam
        beta_mag = numerator / denom
        if lam < beta_mag <= a * lam:
            candidates.add(sign * beta_mag)

    # Region 3: no shrinkage.
    if abs_z > a * lam:
        candidates.add(z)

    # Boundary points ensure we consider potential minima.
    candidates.add(sign * lam)
    candidates.add(sign * a * lam)

    def objective(beta: float) -> float:
        return 0.5 * (beta - z) ** 2 + step_size * _scad_penalty(abs(beta), lam, a)

    best_val = float("inf")
    best_beta = 0.0
    for beta in candidates:
        val = objective(beta)
        if val < best_val:
            best_val = val
            best_beta = beta
    return best_beta


def _scad_penalty(abs_beta: float, lam: float, a: float) -> float:
    if abs_beta <= lam:
        return lam * abs_beta
    if abs_beta <= a * lam:
        return (-(abs_beta**2) + 2 * a * lam * abs_beta - lam**2) / (2 * (a - 1))
    return 0.5 * (a + 1) * lam**2


__all__ = [
    "BaseRegularizer",
    "L1Regularizer",
    "L2Regularizer",
    "NoRegularizer",
    "SCADRegularizer",
]
