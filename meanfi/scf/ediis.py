from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EDIISPoint:
    """One density and its exact quadratic-functional decomposition."""

    params: np.ndarray
    one_body_energy: float
    energy: float


def _recent_lowest_energy_index(history: Sequence[EDIISPoint]) -> int:
    energies = np.asarray([point.energy for point in history], dtype=float)
    minimum = int(np.argmin(energies))
    tied = np.flatnonzero(
        np.isclose(energies, energies[minimum], rtol=1e-12, atol=1e-14)
    )
    return int(tied[-1])


def ediis_coefficients(
    history: Sequence[EDIISPoint],
    *,
    interaction_energy: Callable[[np.ndarray], float],
    interaction_gradient: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Minimize the exact quadratic energy over the convex history hull."""

    count = len(history)
    if count == 0:
        raise ValueError("EDIIS history must be non-empty")
    if count == 1:
        return np.ones(1, dtype=float)

    params = np.stack([point.params for point in history])
    one_body = np.asarray([point.one_body_energy for point in history], dtype=float)

    def objective(coefficients: np.ndarray) -> float:
        mixed = np.tensordot(coefficients, params, axes=1)
        return float(coefficients @ one_body + interaction_energy(mixed))

    def gradient(coefficients: np.ndarray) -> np.ndarray:
        mixed = np.tensordot(coefficients, params, axes=1)
        return one_body + np.asarray(
            [interaction_gradient(mixed, direction) for direction in params],
            dtype=float,
        )

    starts = [np.full(count, 1.0 / count)]
    starts.extend(np.eye(count, dtype=float))
    best = None
    best_value = float("inf")
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints={
                "type": "eq",
                "fun": lambda coefficients: float(np.sum(coefficients) - 1.0),
                "jac": lambda coefficients: np.ones_like(coefficients),
            },
            options={"ftol": 1e-12, "maxiter": 200},
        )
        coefficients = np.asarray(result.x, dtype=float)
        feasible = (
            bool(result.success)
            and np.all(np.isfinite(coefficients))
            and np.min(coefficients) >= -1e-9
            and abs(float(np.sum(coefficients)) - 1.0) <= 1e-8
        )
        if not feasible:
            continue
        coefficients = np.clip(coefficients, 0.0, 1.0)
        coefficients /= np.sum(coefficients)
        value = objective(coefficients)
        if value < best_value:
            best = coefficients
            best_value = value

    recent_best = _recent_lowest_energy_index(history)
    vertex = np.zeros(count, dtype=float)
    vertex[recent_best] = 1.0
    if best is None:
        return vertex

    recent = history[recent_best]
    if best_value >= recent.energy:
        return vertex
    return best


__all__ = ["EDIISPoint", "ediis_coefficients"]
