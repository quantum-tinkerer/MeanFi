from __future__ import annotations

import numpy as np

from meanfi.core.matrix import as_sparse, is_sparse_like
from meanfi.core.results import SolverResult
from meanfi.integrate.methods import IntegrationMethod
from meanfi.params.rparams import bdg_tb_to_rparams, canonical_tb_keys, rparams_to_bdg_tb
from meanfi.scf.engine import (
    SCFRunState,
    build_scf_info,
    max_norm,
    record_density_result,
    solve_fixed_point,
)
from meanfi.scf.methods import SCFMethod

from .bdg import (
    bdg_correction_from_density,
    bdg_density_keys,
    validate_bdg_tb,
    zero_bdg_array,
)
from .density import solve_bdg_density_fixed_filling

def _fill_bdg_support(model, tb, support_keys: list[tuple[int, ...]]):
    zero = zero_bdg_array(model._ndof)
    use_sparse = any(is_sparse_like(value) for value in tb.values())
    if use_sparse:
        zero_sparse = as_sparse(zero)
        return {
            key: as_sparse(tb.get(key, zero_sparse))
            for key in support_keys
        }
    return {
        key: np.asarray(tb.get(key, zero), dtype=complex)
        for key in support_keys
    }


def solve_bdg_scf(
    model,
    guess,
    *,
    integration: IntegrationMethod,
    scf: SCFMethod,
    scf_tol: float,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
):
    if scf_tol <= 0:
        raise ValueError("scf_tol must be positive")

    validate_bdg_tb(
        guess,
        ndof=model._ndof,
        ndim=model._ndim,
        name="BdG correction",
    )

    onsite = (0,) * model._ndim
    support_keys = canonical_tb_keys(set(guess) | {onsite})
    density_keys = bdg_density_keys(model, guess)
    initial_meanfield = _fill_bdg_support(model, guess, support_keys)
    params0 = np.asarray(bdg_tb_to_rparams(initial_meanfield, model._ndof), dtype=float)
    state = SCFRunState()

    def residual_fn(params: np.ndarray) -> np.ndarray:
        meanfield = rparams_to_bdg_tb(params, support_keys, model._ndof)
        density_result = solve_bdg_density_fixed_filling(
            model,
            meanfield,
            keys=density_keys,
            integration=integration,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_guess=state.mu,
        )
        record_density_result(state, density_result)

        updated = _fill_bdg_support(
            model,
            bdg_correction_from_density(density_result.density_matrix, model),
            support_keys,
        )
        validate_bdg_tb(
            updated,
            ndof=model._ndof,
            ndim=model._ndim,
            name="BdG correction",
        )
        residual = np.asarray(bdg_tb_to_rparams(updated, model._ndof) - params, dtype=float)
        state.residual_norm = max_norm(residual)
        return residual

    def on_iteration(iteration: int, residual_norm: float) -> None:
        state.iterations = iteration
        state.residual_norm = residual_norm

    result_params = solve_fixed_point(
        residual_fn,
        params0,
        scf=scf,
        scf_tol=scf_tol,
        on_iteration=on_iteration,
    )

    meanfield = rparams_to_bdg_tb(result_params, support_keys, model._ndof)
    density_matrix_result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=density_keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=state.mu,
    )
    residual_norm = max_norm(
        np.asarray(
            bdg_tb_to_rparams(
                _fill_bdg_support(
                    model,
                    bdg_correction_from_density(density_matrix_result.density_matrix, model),
                    support_keys,
                ),
                model._ndof,
            )
            - result_params,
            dtype=float,
        )
    )

    info = build_scf_info(
        state,
        final_result=density_matrix_result,
        scf=scf,
        residual_norm=residual_norm,
    )
    return SolverResult(
        mf=meanfield,
        density_matrix_result=density_matrix_result,
        integration=integration,
        scf=scf,
        info=info,
    )
