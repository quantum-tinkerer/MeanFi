"""Unified requested and achieved numerical errors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ErrorTolerances:
    """Requested scalar tolerances for one density or SCF calculation."""

    scf_residual: float
    density_matrix_integration: float
    filling_residual: float
    charge_integration: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            number = float(value)
            if not np.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, number)


@dataclass(frozen=True)
class ErrorValues:
    """Achieved scalar errors; ``None`` means unavailable or inapplicable."""

    scf_residual: float | None = None
    density_matrix_integration: float | None = None
    filling_residual: float | None = None
    charge_integration: float | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is None:
                continue
            number = float(value)
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be a non-negative finite number")
            object.__setattr__(self, name, number)


ToleranceFunction = Callable[[float], ErrorTolerances]


def default_solver_tolerances(tol: float) -> ErrorTolerances:
    """Map one user tolerance to MeanFi's default error hierarchy."""

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")
    return ErrorTolerances(
        scf_residual=tol,
        density_matrix_integration=tol / 5.0,
        filling_residual=tol / 10.0,
        charge_integration=tol / 5.0,
    )


def resolve_error_tolerances(
    tol: float,
    tolerance_policy: ToleranceFunction,
) -> ErrorTolerances:
    """Evaluate and validate an ordinary ``tol -> tolerances`` callable."""

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")
    if not callable(tolerance_policy):
        raise TypeError("tolerance_policy must be callable")
    tolerances = tolerance_policy(tol)
    if not isinstance(tolerances, ErrorTolerances):
        raise TypeError("tolerance_policy must return ErrorTolerances")
    return tolerances


def resolve_integration_tolerances(integration, tolerances: ErrorTolerances):
    """Fill automatic integration tolerances and retain explicit overrides."""

    if getattr(integration, "nk", None) is not None:
        return integration, tolerances

    density_setting = getattr(integration, "density_matrix_tol", None)
    charge_setting = getattr(integration, "charge_tol", None)
    density_tolerance = (
        tolerances.density_matrix_integration
        if density_setting is None
        else float(density_setting)
    )
    charge_tolerance = (
        tolerances.charge_integration
        if charge_setting is None
        else float(charge_setting)
    )
    resolved_tolerances = replace(
        tolerances,
        density_matrix_integration=density_tolerance,
        charge_integration=charge_tolerance,
    )
    return (
        replace(
            integration,
            density_matrix_tol=density_tolerance,
            charge_tol=charge_tolerance,
        ),
        resolved_tolerances,
    )


def density_matrix_error_value(error_matrices) -> float | None:
    """Reduce detailed density errors using the public max-element norm."""

    if error_matrices is None:
        return None
    maxima = [
        float(np.max(np.abs(np.asarray(matrix))))
        for matrix in error_matrices.values()
        if np.asarray(matrix).size
    ]
    if not maxima:
        return 0.0
    return max(maxima)
