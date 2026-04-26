from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.core.results import DensityMatrixResult, SCFInfo, SolverResult
from meanfi.integrate.dispatch import solve_density_matrix_fixed_filling
from meanfi.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.normal.meanfield import meanfield
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.scf import LinearMixing, NoConvergence, SCFMethod, max_norm, solve_fixed_point
from meanfi.superconducting.scf import solve_bdg_scf
from meanfi.tb.tb import _tb_type


@dataclass
class _ScfState:
    iterations: int = 0
    mu: float = 0.0
    density_matrix_result: DensityMatrixResult | None = None
    residual_norm: float = float("inf")
    total_charge_integration_calls: int = 0
    total_density_integration_calls: int = 0
    total_kernel_evals: int = 0
    total_unique_evals: int = 0
    total_evaluator_evals: int = 0


def _integration_counters(result: DensityMatrixResult) -> tuple[int, int, int, int, int]:
    info = result.info
    return (
        int(getattr(info, "charge_integration_calls", 0) or 0),
        int(getattr(info, "density_integration_calls", 0) or 0),
        int(getattr(info, "n_kernel_evals", 0) or 0),
        int(
            getattr(
                info,
                "unique_evals",
                getattr(info, "n_kernel_evals", getattr(info, "n_kpoints", 0)),
            )
            or 0
        ),
        int(getattr(info, "n_evaluator_evals", 0) or 0),
    )


def _record_density_result(state: _ScfState, result: DensityMatrixResult) -> None:
    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = _integration_counters(
        result
    )
    state.density_matrix_result = result
    state.mu = result.mu
    state.total_charge_integration_calls += charge_calls
    state.total_density_integration_calls += density_calls
    state.total_kernel_evals += kernel_evals
    state.total_unique_evals += unique_evals
    state.total_evaluator_evals += evaluator_evals


def _density_for_hamiltonian(
    model: Model,
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
) -> DensityMatrixResult:
    return solve_density_matrix_fixed_filling(
        hamiltonian,
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=mu_guess,
    )


def solver(
    model: Model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod,
    scf: SCFMethod = LinearMixing(),
    scf_tol: float = 1e-5,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_mu_iterations: int | None = None,
) -> SolverResult:
    """Solve for the self-consistent mean-field correction."""

    if scf_tol <= 0:
        raise ValueError("scf_tol must be positive")
    if getattr(model, "superconducting", False):
        return solve_bdg_scf(
            model,
            guess,
            integration=integration,
            scf=scf,
            scf_tol=scf_tol,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
        )

    keys = list(model.h_int)
    density_matrix_result = _density_for_hamiltonian(
        model,
        model.hamiltonian_from_meanfield(guess),
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=0.0,
    )
    density_guess = {key: density_matrix_result.density_matrix[key] for key in keys}
    density_params0 = tb_to_rparams(density_guess)

    state = _ScfState(mu=density_matrix_result.mu, density_matrix_result=density_matrix_result)
    _record_density_result(state, density_matrix_result)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        density_guess = rparams_to_tb(params, keys, model._ndof)
        density_result = _density_for_hamiltonian(
            model,
            model.hamiltonian_from_rho(density_guess),
            keys=keys,
            integration=integration,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_guess=state.mu,
        )
        density_new = {key: density_result.density_matrix[key] for key in keys}
        residual = np.asarray(tb_to_rparams(density_new) - params, dtype=float).real
        _record_density_result(state, density_result)
        state.residual_norm = max_norm(residual)
        return residual

    def on_iteration(iteration: int, residual_norm: float) -> None:
        state.iterations = iteration
        state.residual_norm = residual_norm

    result_params = solve_fixed_point(
        residual_fn,
        density_params0,
        scf=scf,
        scf_tol=scf_tol,
        on_iteration=on_iteration,
    )

    density_result_guess = rparams_to_tb(result_params, keys, model._ndof)
    density_matrix_result = _density_for_hamiltonian(
        model,
        model.hamiltonian_from_rho(density_result_guess),
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=state.mu,
    )
    density_reduced = {
        key: density_matrix_result.density_matrix[key]
        for key in keys
    }
    residual_norm = max_norm(tb_to_rparams(density_reduced) - result_params)
    mf_result = meanfield(density_reduced, model.h_int)
    tb_result = dict(mf_result)
    tb_result[model._local_key] = tb_result.get(
        model._local_key,
        np.zeros((model._ndof, model._ndof), dtype=complex),
    ) - density_matrix_result.mu * np.eye(model._ndof)

    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = _integration_counters(
        density_matrix_result
    )
    info = SCFInfo(
        method="anderson_mixing" if scf.__class__.__name__ == "AndersonMixing" else "linear_mixing",
        iterations=max(1, state.iterations),
        residual_norm=residual_norm,
        total_charge_integration_calls=state.total_charge_integration_calls
        + charge_calls,
        total_density_integration_calls=state.total_density_integration_calls
        + density_calls,
        total_kernel_evals=state.total_kernel_evals + kernel_evals,
        total_unique_evals=state.total_unique_evals + unique_evals,
        total_evaluator_evals=state.total_evaluator_evals + evaluator_evals,
    )
    return SolverResult(
        mf=tb_result,
        density_matrix_result=density_matrix_result,
        integration=integration,
        scf=scf,
        info=info,
    )


__all__ = ["NoConvergence", "solver"]
