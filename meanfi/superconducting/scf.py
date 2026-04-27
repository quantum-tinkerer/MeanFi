from __future__ import annotations

import numpy as np

from meanfi.core.matrix import as_sparse, is_sparse_like
from meanfi.core.results import DensityMatrixResult, SolverResult
from meanfi.integrate.methods import IntegrationMethod
from meanfi.params.rparams import bdg_tb_to_rparams, canonical_tb_keys, rparams_to_bdg_tb
from meanfi.scf.engine import SCFRunState, build_scf_info, run_scf_problem
from meanfi.scf.methods import SCFMethod

from .bdg import bdg_correction_from_density, bdg_density_keys, validate_bdg_tb, zero_bdg_array
from .density import solve_bdg_density_fixed_filling


def _fill_bdg_support(model, tb, support_keys: list[tuple[int, ...]]):
    zero = zero_bdg_array(model._ndof)
    use_sparse = any(is_sparse_like(value) for value in tb.values())
    if use_sparse:
        zero_sparse = as_sparse(zero)
        return {key: as_sparse(tb.get(key, zero_sparse)) for key in support_keys}
    return {key: np.asarray(tb.get(key, zero), dtype=complex) for key in support_keys}


def _validated_bdg_update(model, support_keys, density_matrix) -> dict[tuple[int, ...], np.ndarray]:
    updated = _fill_bdg_support(
        model,
        bdg_correction_from_density(density_matrix, model),
        support_keys,
    )
    validate_bdg_tb(
        updated,
        ndof=model._ndof,
        ndim=model._ndim,
        name="BdG correction",
    )
    return updated


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

    def evaluate_density(params: np.ndarray, mu_guess: float) -> DensityMatrixResult:
        return solve_bdg_density_fixed_filling(
            model,
            rparams_to_bdg_tb(params, support_keys, model._ndof),
            keys=density_keys,
            integration=integration,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_guess=mu_guess,
        )

    def residual_from_density(
        params: np.ndarray,
        density_result: DensityMatrixResult,
    ) -> np.ndarray:
        updated = _validated_bdg_update(
            model,
            support_keys,
            density_result.density_matrix,
        )
        return np.asarray(bdg_tb_to_rparams(updated, model._ndof) - params, dtype=float)

    run = run_scf_problem(
        params0,
        evaluate_density=evaluate_density,
        residual_from_density=residual_from_density,
        scf=scf,
        scf_tol=scf_tol,
        state=SCFRunState(),
    )

    meanfield = rparams_to_bdg_tb(run.params, support_keys, model._ndof)
    density_matrix_result = run.final_density_result
    info = build_scf_info(
        run.state,
        final_result=density_matrix_result,
        scf=scf,
        residual_norm=run.residual_norm,
    )
    return SolverResult(
        mf=meanfield,
        density_matrix_result=density_matrix_result,
        integration=integration,
        scf=scf,
        info=info,
    )
