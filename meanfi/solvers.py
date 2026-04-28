from __future__ import annotations

import inspect

import numpy as np

from meanfi.core.results import DensityMatrixResult, SolverResult
from meanfi.integrate.density_support import normal_density_entry_support
from meanfi.integrate.dispatch import solve_density_matrix_fixed_filling
from meanfi.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.normal.meanfield import meanfield
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.scf.engine import NoConvergence, SCFRunState, build_scf_info, record_density_result, run_scf_problem
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
    density_entry_support=None,
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
        density_entry_support=density_entry_support,
    )


def _evaluate_density_for_hamiltonian(
    model: Model,
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
    density_entry_support=None,
) -> DensityMatrixResult:
    kwargs = dict(
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=mu_guess,
    )
    if "density_entry_support" in inspect.signature(_density_for_hamiltonian).parameters:
        kwargs["density_entry_support"] = density_entry_support
    return _density_for_hamiltonian(model, hamiltonian, **kwargs)


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
    density_support = normal_density_entry_support(
        keys=keys if model._local_key in keys else [*keys, model._local_key],
        interaction_support=model.h_int,
        ndof=model._ndof,
        local_key=model._local_key,
    )
    initial_density_result = _evaluate_density_for_hamiltonian(
        model,
        model.hamiltonian_from_meanfield(guess),
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=0.0,
        density_entry_support=density_support,
    )
    density_guess = {key: initial_density_result.density_matrix[key] for key in keys}
    density_params0 = tb_to_rparams(density_guess)

    state = SCFRunState()
    record_density_result(state, initial_density_result)

    def evaluate_density(params: np.ndarray, mu_guess: float) -> DensityMatrixResult:
        return _evaluate_density_for_hamiltonian(
            model,
            model.hamiltonian_from_rho(rparams_to_tb(params, keys, model._ndof)),
            keys=keys,
            integration=integration,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_guess=mu_guess,
            density_entry_support=density_support,
        )

    def residual_from_density(
        params: np.ndarray,
        density_result: DensityMatrixResult,
    ) -> np.ndarray:
        density_new = {key: density_result.density_matrix[key] for key in keys}
        return np.asarray(tb_to_rparams(density_new) - params, dtype=float).real

    run = run_scf_problem(
        density_params0,
        evaluate_density=evaluate_density,
        residual_from_density=residual_from_density,
        scf=scf,
        scf_tol=scf_tol,
        state=state,
    )

    density_matrix_result = run.final_density_result
    density_reduced = {key: density_matrix_result.density_matrix[key] for key in keys}
    mf_result = meanfield(density_reduced, model.h_int)
    tb_result = dict(mf_result)
    tb_result[model._local_key] = tb_result.get(
        model._local_key,
        np.zeros((model._ndof, model._ndof), dtype=complex),
    ) - density_matrix_result.mu * np.eye(model._ndof)

    info = build_scf_info(
        run.state,
        final_result=density_matrix_result,
        scf=scf,
        residual_norm=run.residual_norm,
    )
    return SolverResult(
        mf=tb_result,
        density_matrix_result=density_matrix_result,
        integration=integration,
        scf=scf,
        info=info,
    )


__all__ = ["NoConvergence", "solver"]
