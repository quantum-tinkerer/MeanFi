"""Top-level density-evaluation pipeline.

This file intentionally mirrors the algorithm overview. Public APIs build a
problem, choose a numerical plan, evaluate either a fixed chemical potential or
the fixed-filling nested solve, and then return the public result object.
"""

from __future__ import annotations

from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.density.integrate.bdg import solve_bdg_density_fixed_filling
from meanfi.density.plan import (
    build_plan,
    evaluate_at_mu,
    evaluate_fixed_filling,
)
from meanfi.density.problem import (
    DensityEvaluation,
    DensityPlan,
    build_normal_problem,
)
from meanfi.density.results import (
    wrap_density_evaluation,
)
from meanfi.results import DensityMatrixResult
from meanfi.space.density_selection import DensitySelection
from meanfi.tb.ops import _tb_type


def solve_density_matrix_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    density_selection: DensitySelection | None = None,
) -> DensityMatrixResult:
    problem = build_normal_problem(
        hamiltonian,
        kT=kT,
        keys=keys,
        integration=integration,
        density_selection=density_selection,
    )
    plan = build_plan(problem)
    evaluation = evaluate_at_mu(problem, plan, mu)
    return wrap_density_evaluation(problem, plan, evaluation)


def solve_density_matrix_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float = 0.0,
    density_selection: DensitySelection | None = None,
) -> DensityMatrixResult:
    problem = build_normal_problem(
        hamiltonian,
        kT=kT,
        keys=keys,
        integration=integration,
        density_selection=density_selection,
    )
    plan = build_plan(problem)
    evaluation = evaluate_fixed_filling(
        problem,
        plan,
        filling=filling,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_guess=mu_guess,
    )
    return wrap_density_evaluation(problem, plan, evaluation, target_filling=filling)


__all__ = [
    "DensityEvaluation",
    "DensityPlan",
    "solve_bdg_density_fixed_filling",
    "solve_density_matrix_at_mu",
    "solve_density_matrix_fixed_filling",
]
