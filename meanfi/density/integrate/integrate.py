"""Choose one of the two Brillouin-zone integration families."""

from meanfi.density.integrate.methods import AdaptiveSimplex, PeriodicGrid
from meanfi.density.integrate.normal import (
    _adaptive_simplex_at_mu,
    _adaptive_simplex_fixed_filling,
)
from meanfi.density.integrate.periodic import solve_periodic
from meanfi.density.problem import DensityPlan, DensityProblem


def build_integration_plan(problem: DensityProblem) -> DensityPlan:
    if isinstance(problem.integration, AdaptiveSimplex):
        return DensityPlan(
            integration=problem.integration,
            evaluate_mu=lambda mu: _adaptive_simplex_at_mu(problem, mu),
            solve_filling=lambda filling,
            filling_tol,
            mu_tol,
            max_evals,
            mu_guess: _adaptive_simplex_fixed_filling(
                problem, filling, filling_tol, mu_tol, max_evals, mu_guess
            ),
        )
    if not isinstance(problem.integration, PeriodicGrid):
        raise TypeError("integration must be AdaptiveSimplex or PeriodicGrid")

    def evaluate(**kwargs):
        return solve_periodic(
            problem.hamiltonian,
            kT=problem.kT,
            keys=problem.solve_keys,
            integration=problem.integration,
            density_coordinates=problem.density_coordinates,
            tolerances=problem.tolerances,
            **kwargs,
        )

    return DensityPlan(
        integration=problem.integration,
        evaluate_mu=lambda mu: evaluate(mu=mu),
        solve_filling=lambda filling,
        filling_tol,
        mu_tol,
        max_evals,
        mu_guess: evaluate(
            filling=filling,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_charge_evaluations=max_evals,
            mu_guess=mu_guess,
        ),
    )
