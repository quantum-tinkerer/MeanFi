from __future__ import annotations

from meanfi.core.results import DensityMatrixResult
from meanfi.integrate.dispatch import solve_density_matrix_at_mu, solve_density_matrix_fixed_filling
from meanfi.integrate.methods import IntegrationMethod
from meanfi.integrate.quadrature.normal_backend import fermi_dirac
from meanfi.normal.meanfield import meanfield

__all__ = [
    "DensityMatrixResult",
    "density_matrix",
    "density_matrix_at_mu",
    "fermi_dirac",
    "meanfield",
]


def density_matrix_at_mu(
    h,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    *,
    integration: IntegrationMethod,
) -> DensityMatrixResult:
    """Compute the real-space density matrix at a fixed chemical potential."""

    return solve_density_matrix_at_mu(
        h,
        mu=mu,
        kT=kT,
        keys=keys,
        integration=integration,
    )


def density_matrix(
    h,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    *,
    integration: IntegrationMethod,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_mu_iterations: int | None = None,
) -> DensityMatrixResult:
    """Compute the fixed-filling real-space density matrix."""

    return solve_density_matrix_fixed_filling(
        h,
        filling=filling,
        kT=kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
    )
