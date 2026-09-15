"""Unified requested and achieved numerical errors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np


class ConvergenceError(RuntimeError):
    """A density integration or filling solve could not meet its numerical target."""


@dataclass(frozen=True)
class ErrorTolerances:
    """Absolute targets for density, filling, and SCF convergence."""

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
    band_energy_integration: float | None = None
    entropy_integration: float | None = None

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
    """An explicit density target also supplies an omitted charge target.

    Without mesh overrides, retain both targets from the tolerance policy.
    """
    charge = integration.charge_tol
    if charge is None and integration.density_matrix_tol is not None:
        charge = integration.density_matrix_tol
    settings = {
        "density_matrix_integration": integration.density_matrix_tol,
        "charge_integration": charge,
    }
    return replace(tolerances, **{k: v for k, v in settings.items() if v is not None})
