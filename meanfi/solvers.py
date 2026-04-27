from __future__ import annotations

import numpy as np

from meanfi.core.results import DensityMatrixResult, SolverResult
from meanfi.integrate.dispatch import solve_density_matrix_fixed_filling
from meanfi.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.normal.meanfield import meanfield
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.scf.engine import (
    NoConvergence,
    SCFRunState,
    build_scf_info,
    max_norm,
    record_density_result,
    solve_fixed_point,
)
from meanfi.scf.methods import LinearMixing, SCFMethod
from meanfi.superconducting.scf import solve_bdg_scf
from meanfi.tb.tb import _tb_type


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

    state = SCFRunState()
    record_density_result(state, density_matrix_result)

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
        record_density_result(state, density_result)
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

    info = build_scf_info(
        state,
        final_result=density_matrix_result,
        scf=scf,
        residual_norm=residual_norm,
    )
    return SolverResult(
        mf=tb_result,
        density_matrix_result=density_matrix_result,
        integration=integration,
        scf=scf,
        info=info,
    )


__all__ = ["NoConvergence", "solver"]
