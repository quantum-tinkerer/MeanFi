from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EDIISPoint:
    """One evaluated density and its internal energy."""

    params: np.ndarray
    internal_energy: float


def ediis_coefficients(
    history: Sequence[EDIISPoint],
    *,
    interaction_gradient: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Minimize internal energy over the convex density history.

    The one-body term is linear in density and the interaction is quadratic,
    so this is the mixed density's internal energy at every temperature.
    """

    count = len(history)
    if count == 0:
        raise ValueError("EDIIS history must be non-empty")
    if count == 1:
        return np.ones(1, dtype=float)

    energies = np.asarray([point.internal_energy for point in history], dtype=float)
    curvature = np.zeros((count, count))
    for i, left in enumerate(history):
        for j, right in enumerate(history[:i]):
            difference = left.params - right.params
            curvature[i, j] = curvature[j, i] = 0.5 * (
                interaction_gradient(left.params, difference)
                - interaction_gradient(right.params, difference)
            )

    # For a quadratic energy and sum(c) = 1, this equals E(sum(c_i rho_i)).
    # Prepare the small history matrix once; optimization needs no model calls.
    def objective(coefficients: np.ndarray) -> float:
        return float(
            coefficients @ energies - 0.5 * coefficients @ curvature @ coefficients
        )

    def gradient(coefficients: np.ndarray) -> np.ndarray:
        return energies - curvature @ coefficients

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

    recent_best = np.flatnonzero(
        np.isclose(energies, np.min(energies), rtol=1e-12, atol=1e-14)
    )[-1]
    vertex = np.zeros(count, dtype=float)
    vertex[recent_best] = 1.0
    if best is None:
        return vertex

    if best_value >= energies[recent_best]:
        return vertex
    return best


__all__ = ["EDIISPoint", "ediis_coefficients"]
