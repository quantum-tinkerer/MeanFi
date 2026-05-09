"""Numerical plan selection and execution for density solves."""

from __future__ import annotations

from meanfi.density.integrate.integrate import build_integration_plan
from meanfi.density.problem import DensityEvaluation, DensityPlan, DensityProblem


def build_plan(problem: DensityProblem) -> DensityPlan:
    """Build the numerical plan below the density pipeline."""

    return build_integration_plan(problem)


def evaluate_at_mu(
    problem: DensityProblem,
    plan: DensityPlan,
    mu: float,
) -> DensityEvaluation:
    """Evaluate the density once the chemical potential is fixed."""

    del problem
    return DensityEvaluation(result=plan.evaluate_mu(mu))


def evaluate_fixed_filling(
    problem: DensityProblem,
    plan: DensityPlan,
    *,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float = 0.0,
) -> DensityEvaluation:
    """Run the nested chemical-potential solve, then evaluate density."""

    del problem
    if mu_tol <= 0:
        raise ValueError("mu_tol must be positive")
    if max_charge_evaluations is not None and max_charge_evaluations <= 0:
        raise ValueError("max_charge_evaluations must be positive")
    return DensityEvaluation(
        result=plan.solve_filling(
            filling,
            filling_tol,
            mu_tol,
            max_charge_evaluations,
            mu_guess,
        )
    )


__all__ = ["build_plan", "evaluate_at_mu", "evaluate_fixed_filling"]
