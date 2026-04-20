from __future__ import annotations

import numpy as np

from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import tb_to_native_spectral_cache

try:
    from meanfi._zero_temp_native import (
        ChargeSolveOptions,
        DensityIntegrateOptions,
        NativeChargeEvaluator,
        NativeDensityEvaluator,
        NativeFrontier,
        NativeGeometry,
    )

    _NATIVE_ZERO_TEMP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when native extension is unavailable
    ChargeSolveOptions = None
    DensityIntegrateOptions = None
    NativeChargeEvaluator = None
    NativeDensityEvaluator = None
    NativeFrontier = None
    NativeGeometry = None
    _NATIVE_ZERO_TEMP_AVAILABLE = False


_GEOM_TOL = 1e-14
_BULK_THETA = 0.5
_ROOT_SUBCELLS_PER_AXIS = 2


def _root_subcells_per_axis(ndim: int) -> int:
    return 4 if ndim == 1 else _ROOT_SUBCELLS_PER_AXIS


def _require_native_backend() -> None:
    if not _NATIVE_ZERO_TEMP_AVAILABLE or NativeGeometry is None:
        raise RuntimeError(
            "Zero-temperature integration requires the native meanfi._zero_temp_native extension"
        )


def _build_native_runtime(h: _tb_type):
    _require_native_backend()
    ndim = len(next(iter(h)))
    geometry = NativeGeometry.root(
        ndim,
        root_subcells_per_axis=_root_subcells_per_axis(ndim),
        tol=float(_GEOM_TOL),
    )
    frontier = NativeFrontier.from_geometry(geometry)
    spectral_cache = tb_to_native_spectral_cache(h, tol=float(_GEOM_TOL))
    return geometry, frontier, spectral_cache


def _vector_to_density(
    estimate: np.ndarray,
    error: np.ndarray,
    ndof: int,
    keys: list[tuple[int, ...]],
):
    estimate = np.asarray(estimate).reshape(ndof, ndof, len(keys))
    error = np.asarray(error).reshape(ndof, ndof, len(keys))
    rho = {}
    rho_error = {}
    for index, key in enumerate(keys):
        rho[key] = estimate[..., index]
        rho_error[key] = error[..., index]
    return rho, rho_error


def _density_integration_info(*, result, spectral_cache):
    from meanfi.mf import DensityIntegrationInfo

    return DensityIntegrationInfo(
        n_kernel_evals=int(spectral_cache.n_kernel_evals),
        n_evaluator_evals=int(result.evaluator_evals),
        n_cached_nodes=int(spectral_cache.size),
        n_leaves=int(result.n_leaves),
        n_leaf_nodes=int(result.n_leaf_nodes),
        subdivisions=int(result.subdivisions),
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
):
    from meanfi.mf import FixedFillingInfo

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
        charge_n_evaluator_evals=int(charge_result.evaluator_evals),
        density_n_evaluator_evals=int(density_result.evaluator_evals),
        n_evaluator_evals=int(charge_result.evaluator_evals + density_result.evaluator_evals),
        n_cached_nodes=int(spectral_cache.size),
        n_leaves=int(density_result.n_leaves),
        n_leaf_nodes=int(density_result.n_leaf_nodes),
        subdivisions=int(charge_result.subdivisions + density_result.subdivisions),
        charge_integral_atol=float(charge_tol),
        density_atol=float(density_atol),
        density_rtol=float(density_rtol),
    )


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
    max_mu_iterations: int,
    max_subdivisions: int | None,
):
    geometry, frontier, spectral_cache = _build_native_runtime(h)

    charge_options = ChargeSolveOptions()
    charge_options.mu_guess = float(mu_guess)
    charge_options.charge_tol = float(charge_tol)
    charge_options.mu_xtol = float(mu_xtol)
    charge_options.max_mu_iterations = int(max_mu_iterations)
    charge_options.max_subdivisions = -1 if max_subdivisions is None else int(max_subdivisions)
    charge_options.bulk_theta = float(_BULK_THETA)

    charge_evaluator = NativeChargeEvaluator(geometry, spectral_cache, tol=float(_GEOM_TOL))
    try:
        charge_result = charge_evaluator.solve_mu_and_refine(
            frontier,
            float(filling),
            charge_options,
        )
    except RuntimeError as exc:
        if "Adaptive zero-temperature" in str(exc):
            raise ValueError(str(exc)) from exc
        raise

    charge_kernel_evals = int(spectral_cache.n_kernel_evals)

    density_options = DensityIntegrateOptions()
    density_options.density_atol = float(density_atol)
    density_options.density_rtol = float(density_rtol)
    if max_subdivisions is None:
        density_options.max_subdivisions = -1
    else:
        density_options.max_subdivisions = max(
            int(max_subdivisions) - int(charge_result.subdivisions),
            0,
        )
    density_options.bulk_theta = float(_BULK_THETA)

    keys_arr = np.ascontiguousarray(np.asarray(keys, dtype=np.float64))
    density_evaluator = NativeDensityEvaluator(
        geometry,
        spectral_cache,
        keys_arr,
        tol=float(_GEOM_TOL),
    )
    try:
        density_result = density_evaluator.integrate_adaptive(
            frontier,
            float(charge_result.mu),
            density_options,
        )
    except RuntimeError as exc:
        if "Adaptive zero-temperature" in str(exc):
            raise ValueError(str(exc)) from exc
        raise

    density_kernel_evals = int(spectral_cache.n_kernel_evals) - charge_kernel_evals
    rho, error = _vector_to_density(
        np.asarray(density_result.estimate_array(), dtype=complex),
        np.asarray(density_result.error_vector_array(), dtype=float),
        int(spectral_cache.ndof),
        keys,
    )
    info = _fixed_filling_info(
        charge_result=charge_result,
        density_result=density_result,
        spectral_cache=spectral_cache,
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
    max_subdivisions: int | None,
):
    geometry, frontier, spectral_cache = _build_native_runtime(h)

    density_options = DensityIntegrateOptions()
    density_options.density_atol = float(density_atol)
    density_options.density_rtol = float(density_rtol)
    density_options.max_subdivisions = -1 if max_subdivisions is None else int(max_subdivisions)
    density_options.bulk_theta = float(_BULK_THETA)

    keys_arr = np.ascontiguousarray(np.asarray(keys, dtype=np.float64))
    density_evaluator = NativeDensityEvaluator(
        geometry,
        spectral_cache,
        keys_arr,
        tol=float(_GEOM_TOL),
    )
    try:
        density_result = density_evaluator.integrate_adaptive(
            frontier,
            float(mu),
            density_options,
        )
    except RuntimeError as exc:
        if "Adaptive zero-temperature" in str(exc):
            raise ValueError(str(exc)) from exc
        raise

    rho, error = _vector_to_density(
        np.asarray(density_result.estimate_array(), dtype=complex),
        np.asarray(density_result.error_vector_array(), dtype=float),
        int(spectral_cache.ndof),
        keys,
    )
    info = _density_integration_info(result=density_result, spectral_cache=spectral_cache)
    return rho, error, info
