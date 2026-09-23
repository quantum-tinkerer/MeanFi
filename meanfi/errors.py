"""Unified requested and achieved numerical errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class ConvergenceError(RuntimeError):
    """A density integration or filling solve could not meet its numerical target."""


@dataclass(frozen=True, kw_only=True)
class ErrorTolerances:
    """Absolute targets accepted as ``tol`` by density calculations and SCF.

    Omitted ``charge_integration`` takes ``density_matrix_integration``.
    ``mu_tol`` limits root-search steps; only the filling residual establishes
    charge convergence. Energy and entropy have no accuracy targets.
    """

    scf_residual: float
    density_matrix_integration: float
    filling_residual: float
    matrix_function_tol: float
    charge_integration: float | None = None
    mu_tol: float = 1e-10

    def __post_init__(self) -> None:
        if self.charge_integration is None:
            object.__setattr__(
                self, "charge_integration", self.density_matrix_integration
            )
        for name, value in self.__dict__.items():
            number = float(value)
            if not np.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, number)


@dataclass(frozen=True)
class ErrorValues:
    """Achieved scalar errors; ``None`` means unavailable or inapplicable.

    Entropy combines integration and matrix-function estimates, per physical
    orbital, when both are available. It has no accuracy target. The sampled
    density-cut estimate is separate from the p-quadrature estimate. Adaptive
    FermiSimplex uses it to avoid refining p/h quadrature far below the
    estimated cut error.
    """

    scf_residual: float | None = None
    density_matrix_integration: float | None = None
    # Sampled error from placing the occupation cut on the charge mesh.
    density_cut_estimate: float | None = None
    filling_residual: float | None = None
    charge_integration: float | None = None
    band_energy_integration: float | None = None
    entropy: float | None = None
    matrix_function_error: float | None = None

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
    """Assign the same simple tolerance budgets to every calculation."""
    return ErrorTolerances(
        scf_residual=tol,
        density_matrix_integration=tol / 5,
        charge_integration=tol / 5,
        filling_residual=tol / 10,
        matrix_function_tol=tol / 40,
    )


def resolve_error_tolerances(
    tol: float | ErrorTolerances,
    tolerance_policy: ToleranceFunction,
) -> ErrorTolerances:
    """Accept explicit targets or apply the scalar tolerance policy once."""
    if isinstance(tol, ErrorTolerances):
        if tolerance_policy is not default_solver_tolerances:
            raise ValueError(
                "An ErrorTolerances record cannot be combined with a custom tolerance_policy"
            )
        return tol

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")
    if not callable(tolerance_policy):
        raise TypeError("tolerance_policy must be callable")
    tolerances = tolerance_policy(tol)
    if not isinstance(tolerances, ErrorTolerances):
        raise TypeError("tolerance_policy must return ErrorTolerances")
    return tolerances
