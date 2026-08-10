"""Top-level density-evaluation pipeline."""

from __future__ import annotations

from meanfi.errors import ErrorTolerances
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.density.internal import DensityEvaluation
from meanfi.density.plan import build_plan, evaluate_at_mu, evaluate_fixed_filling
from meanfi.density.problem import DensityPlan, build_normal_problem
from meanfi.density.results import wrap_density_evaluation
from meanfi.results import DensityResult
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import _tb_type


def evaluate_density_matrix_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
    density_coordinates: DensityCoordinates | None = None,
) -> tuple[DensityPlan, DensityEvaluation]:
    """Evaluate density internally without materializing uncomputed entries."""

    problem = build_normal_problem(
        hamiltonian,
        kT=kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
        density_coordinates=density_coordinates,
    )
    plan = build_plan(problem)
    return plan, evaluate_at_mu(problem, plan, mu)


def solve_density_matrix_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
) -> DensityResult:
    """Return a complete public density matrix at fixed chemical potential."""

    problem = build_normal_problem(
        hamiltonian,
        kT=kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
    )
    plan = build_plan(problem)
    evaluation = evaluate_at_mu(problem, plan, mu)
    return wrap_density_evaluation(problem, plan, evaluation)


def evaluate_density_matrix_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float = 0.0,
    density_coordinates: DensityCoordinates | None = None,
    include_band_energy: bool = False,
) -> tuple[DensityPlan, DensityEvaluation]:
    """Evaluate fixed-filling density internally on an explicit layout."""

    problem = build_normal_problem(
        hamiltonian,
        kT=kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
        density_coordinates=density_coordinates,
        include_band_energy=include_band_energy,
    )
    plan = build_plan(problem)
    evaluation = evaluate_fixed_filling(
        problem,
        plan,
        filling=filling,
        filling_tol=problem.tolerances.filling_residual,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_guess=mu_guess,
    )
    return plan, evaluation


def solve_density_matrix_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float = 0.0,
) -> DensityResult:
    """Return a complete public density matrix at fixed filling."""

    problem = build_normal_problem(
        hamiltonian,
        kT=kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
    )
    plan = build_plan(problem)
    evaluation = evaluate_fixed_filling(
        problem,
        plan,
        filling=filling,
        filling_tol=problem.tolerances.filling_residual,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_guess=mu_guess,
    )
    return wrap_density_evaluation(problem, plan, evaluation)


__all__ = [
    "evaluate_density_matrix_at_mu",
    "evaluate_density_matrix_fixed_filling",
    "solve_density_matrix_at_mu",
    "solve_density_matrix_fixed_filling",
]
