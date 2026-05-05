from __future__ import annotations

import numpy as np

from meanfi.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.tb.ops import _tb_type

from .backend import (
    AdaptiveIntegrator,
    _GEOM_TOL,
    build_charge_options,
    build_density_options,
    build_extension_runtime,
    raise_normalized_runtime_error,
)


def _vector_to_density(
    estimate: np.ndarray,
    error: np.ndarray,
    ndof: int,
    keys: list[tuple[int, ...]],
) -> tuple[_tb_type, _tb_type]:
    estimate = np.asarray(estimate).reshape(ndof, ndof, len(keys))
    error = np.asarray(error).reshape(ndof, ndof, len(keys))

    rho = {}
    rho_error = {}
    for index, key in enumerate(keys):
        rho[key] = estimate[..., index]
        rho_error[key] = error[..., index]
    return rho, rho_error


def _density_integration_info(*, result, spectral_cache) -> DensityIntegrationInfo:
    return DensityIntegrationInfo(
        n_kernel_evals=int(spectral_cache.n_kernel_evals),
        unique_evals=int(spectral_cache.n_kernel_evals),
        n_evaluator_evals=int(result.evaluator_evals),
        n_cached_nodes=int(spectral_cache.size),
        n_leaves=int(result.n_leaves),
        n_leaf_nodes=int(result.n_leaf_nodes),
        subdivisions=int(result.subdivisions),
        error_estimate_available=bool(
            getattr(result, "error_estimate_available", True)
        ),
    )


def _fixed_filling_info(
    *,
    charge_result,
    density_result,
    spectral_cache,
    charge_kernel_evals: int,
    density_kernel_evals: int,
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
) -> FixedFillingInfo:
    return FixedFillingInfo(
        mu=float(charge_result.mu),
        charge=float(charge_result.charge),
        charge_error=float(charge_result.charge_error),
        dcharge_dmu=float(charge_result.dcharge_dmu),
        root_iterations=int(charge_result.root_iterations),
        charge_integration_calls=int(charge_result.charge_integration_calls),
        density_integration_calls=1,
        charge_n_kernel_evals=int(charge_kernel_evals),
        density_n_kernel_evals=int(density_kernel_evals),
        n_kernel_evals=int(spectral_cache.n_kernel_evals),
        unique_evals=int(spectral_cache.n_kernel_evals),
        charge_n_evaluator_evals=int(charge_result.evaluator_evals),
        density_n_evaluator_evals=int(density_result.evaluator_evals),
        n_evaluator_evals=int(
            charge_result.evaluator_evals + density_result.evaluator_evals
        ),
        n_cached_nodes=int(spectral_cache.size),
        n_leaves=int(density_result.n_leaves),
        n_leaf_nodes=int(density_result.n_leaf_nodes),
        subdivisions=int(charge_result.subdivisions + density_result.subdivisions),
        charge_integral_atol=float(charge_tol),
        density_atol=float(density_atol),
        density_rtol=float(density_rtol),
        error_estimate_available=bool(
            getattr(charge_result, "error_estimate_available", True)
            and getattr(density_result, "error_estimate_available", True)
        ),
    )


def _nan_density_error_like(rho: _tb_type) -> _tb_type:
    return {
        key: np.full(matrix.shape, np.nan, dtype=float) for key, matrix in rho.items()
    }


def density_matrix_zero_temp(
    h: _tb_type,
    *,
    filling: float,
    keys: list[tuple[int, ...]],
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
    mu_guess: float,
    mu_xtol: float,
    max_mu_iterations: int | None,
    max_subdivisions: int | None = None,
    refinement_depth: int = 0,
):
    """Evaluate the zero-temperature fixed-filling density matrix with the compiled backend."""

    if max_subdivisions == 0:
        from .root_mesh import root_mesh_fixed_filling_zero_temp

        return root_mesh_fixed_filling_zero_temp(
            h,
            filling=filling,
            keys=keys,
            charge_tol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            mu_guess=mu_guess,
            mu_xtol=mu_xtol,
            max_mu_iterations=max_mu_iterations,
        )

    geometry, vertex_cache, preview_depth = build_extension_runtime(
        h,
        refinement_depth=refinement_depth,
    )
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        preview_depth,
        tol=float(_GEOM_TOL),
    )

    try:
        charge_result = integrator.solve_mu_and_refine(
            float(filling),
            build_charge_options(
                mu_guess=mu_guess,
                charge_tol=charge_tol,
                mu_xtol=mu_xtol,
                max_mu_iterations=max_mu_iterations,
                max_subdivisions=max_subdivisions,
            ),
        )
    except RuntimeError as exc:
        raise_normalized_runtime_error(exc)

    charge_kernel_evals = int(vertex_cache.n_kernel_evals)
    try:
        density_result = integrator.integrate_density(
            float(charge_result.mu),
            build_density_options(
                density_atol=density_atol,
                density_rtol=density_rtol,
                max_subdivisions=max_subdivisions,
                consumed_subdivisions=int(charge_result.subdivisions),
            ),
        )
    except RuntimeError as exc:
        raise_normalized_runtime_error(exc)

    density_kernel_evals = int(vertex_cache.n_kernel_evals) - charge_kernel_evals
    rho, error = _vector_to_density(
        np.asarray(density_result.estimate_array(), dtype=complex),
        np.asarray(density_result.error_vector_array(), dtype=float),
        int(vertex_cache.ndof),
        keys,
    )
    info = _fixed_filling_info(
        charge_result=charge_result,
        density_result=density_result,
        spectral_cache=vertex_cache,
        charge_kernel_evals=charge_kernel_evals,
        density_kernel_evals=density_kernel_evals,
        charge_tol=charge_tol,
        density_atol=density_atol,
        density_rtol=density_rtol,
    )
    return rho, error, float(charge_result.mu), info


def density_matrix_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    density_atol: float,
    density_rtol: float,
    max_subdivisions: int | None = None,
    refinement_depth: int = 0,
):
    """Evaluate the zero-temperature density matrix at an explicit chemical potential."""

    if max_subdivisions == 0:
        from .root_mesh import root_mesh_density_at_mu_zero_temp

        return root_mesh_density_at_mu_zero_temp(h, mu=mu, keys=keys)

    geometry, vertex_cache, preview_depth = build_extension_runtime(
        h,
        refinement_depth=refinement_depth,
    )
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        preview_depth,
        tol=float(_GEOM_TOL),
    )
    try:
        density_result = integrator.integrate_density(
            float(mu),
            build_density_options(
                density_atol=density_atol,
                density_rtol=density_rtol,
                max_subdivisions=max_subdivisions,
            ),
        )
    except RuntimeError as exc:
        raise_normalized_runtime_error(exc)

    rho, error = _vector_to_density(
        np.asarray(density_result.estimate_array(), dtype=complex),
        np.asarray(density_result.error_vector_array(), dtype=float),
        int(vertex_cache.ndof),
        keys,
    )
    info = _density_integration_info(result=density_result, spectral_cache=vertex_cache)
    return rho, error, info
