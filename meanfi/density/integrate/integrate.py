"""General Brillouin-zone integration entrypoint.

This layer chooses how the density contribution is assembled across k-space.
Scheme-specific modules below it know quadrature, simplex, and uniform-grid
details; the density pipeline above it only sees a plan with two operations.
"""

from __future__ import annotations

from meanfi.density.integrate.methods import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    UniformGrid,
)
from meanfi.density.integrate.normal import (
    _adaptive_quadrature_at_mu,
    _adaptive_quadrature_fixed_filling,
    _adaptive_simplex_at_mu,
    _adaptive_simplex_fixed_filling,
    _uniform_grid_at_mu,
    _uniform_grid_fixed_filling,
)
from meanfi.density.problem import DensityPlan, DensityProblem


def build_integration_plan(problem: DensityProblem) -> DensityPlan:
    """Choose the concrete BZ integration family for one density problem."""

    if problem.family != "normal":
        raise ValueError(f"unsupported density problem family: {problem.family!r}")
    if isinstance(problem.integration, AdaptiveQuadrature):
        solve_at_mu = _adaptive_quadrature_at_mu
        solve_fixed_filling = _adaptive_quadrature_fixed_filling
    elif isinstance(problem.integration, AdaptiveSimplex):
        solve_at_mu = _adaptive_simplex_at_mu
        solve_fixed_filling = _adaptive_simplex_fixed_filling
    elif isinstance(problem.integration, UniformGrid):
        solve_at_mu = _uniform_grid_at_mu
        solve_fixed_filling = _uniform_grid_fixed_filling
    else:
        raise TypeError("integration must be an IntegrationMethod instance")
    return DensityPlan(
        integration=problem.integration,
        evaluate_mu=lambda mu: solve_at_mu(problem, mu),
        solve_filling=lambda filling, filling_tol, mu_tol, max_evals, mu_guess: (
            solve_fixed_filling(
                problem,
                filling,
                filling_tol,
                mu_tol,
                max_evals,
                mu_guess,
            )
        ),
    )


__all__ = ["build_integration_plan"]
