"""Normal-state FermiSimplex dispatch, including a finite-system calculation."""

from __future__ import annotations

from meanfi.density.integrate.simplex import (
    density_matrix_at_mu_zero_temp,
    density_matrix_zero_temp,
)
from meanfi.density.kpoint.zero_dim import evaluate_zero_dim
from meanfi.results import DensityResult
from meanfi.density.problem import DensityProblem
from meanfi.tb.ops import to_dense
from meanfi.tb.validate import tb_dimension


def evaluate_simplex(
    problem: DensityProblem,
    *,
    mu: float | None,
    filling: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
) -> DensityResult:
    integration = problem.integration
    coordinates = problem.density_coordinates
    if tb_dimension(problem.hamiltonian) == 0:
        return evaluate_zero_dim(
            to_dense(problem.hamiltonian[()]),
            coordinates,
            mu=mu,
            filling=filling,
            mu_guess=mu_guess,
            filling_tol=problem.tolerances.filling_residual,
            nk=integration.nk,
        )
    settings = dict(
        keys=list(coordinates.keys),
        density_coordinates=coordinates,
        density_atol=problem.tolerances.density_matrix_integration,
        density_rtol=0.0,
        charge_tol=problem.tolerances.charge_integration,
        max_subdivisions=integration.max_refinements,
        num_threads=integration.num_threads,
        nk=integration.nk,
        max_points=integration.max_points,
    )
    if filling is None:
        return density_matrix_at_mu_zero_temp(problem.hamiltonian, mu=mu, **settings)
    return density_matrix_zero_temp(
        problem.hamiltonian,
        filling=filling,
        filling_tol=problem.tolerances.filling_residual,
        mu_guess=mu_guess,
        mu_xtol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        **settings,
    )
