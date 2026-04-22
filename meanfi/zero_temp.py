from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from meanfi._finite_temp import expand_mu_bracket, mu_bracket, solve_mu
from meanfi._info import DensityIntegrationInfo, FixedFillingInfo
from meanfi.tb._backend import tb_to_vertex_cache
from meanfi.tb.tb import _tb_type

try:
    from meanfi._zero_temp_ext import (
        AdaptiveIntegrator,
        ChargeSolveOptions,
        DensityIntegrateOptions,
        Geometry,
    )

    _ZERO_TEMP_EXT_AVAILABLE = True
except (
    ImportError
):  # pragma: no cover - exercised only when the extension is unavailable
    AdaptiveIntegrator = None
    ChargeSolveOptions = None
    DensityIntegrateOptions = None
    Geometry = None
    _ZERO_TEMP_EXT_AVAILABLE = False


_GEOM_TOL = 1e-14
_BULK_THETA = 0.5
_ROOT_SUBCELLS_PER_AXIS = 2


def _root_subcells_per_axis(ndim: int) -> int:
    return 4 if ndim == 1 else _ROOT_SUBCELLS_PER_AXIS


def _require_zero_temp_extension() -> None:
    if not _ZERO_TEMP_EXT_AVAILABLE or Geometry is None:
        raise RuntimeError(
            "Zero-temperature integration requires the compiled meanfi._zero_temp_ext extension"
        )


def _extension_subdivision_limit(max_subdivisions: int | None) -> int:
    return -1 if max_subdivisions is None else int(max_subdivisions)


def _build_extension_runtime(hamiltonian: _tb_type):
    _require_zero_temp_extension()
    ndim = len(next(iter(hamiltonian)))
    geometry = Geometry.root(
        ndim,
        root_subcells_per_axis=_root_subcells_per_axis(ndim),
        tol=float(_GEOM_TOL),
    )
    vertex_cache = tb_to_vertex_cache(hamiltonian, tol=float(_GEOM_TOL))
    return geometry, vertex_cache


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


def _coarse_density_summary(
    *,
    geometry,
    vertex_cache,
    keys: list[tuple[int, ...]],
    mu: float,
):
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        tol=float(_GEOM_TOL),
    )
    estimate, _owner_ids, _owner_estimates, evaluator_evals = integrator.evaluate_density(
        float(mu),
        0,
    )
    estimate = np.asarray(estimate, dtype=complex)
    zero_error = np.zeros(estimate.shape, dtype=float)
    rho, _ = _vector_to_density(estimate, zero_error, int(vertex_cache.ndof), keys)
    return rho, _nan_density_error_like(rho), int(evaluator_evals)


def _root_mesh_density_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
):
    geometry, vertex_cache = _build_extension_runtime(h)
    rho, error, evaluator_evals = _coarse_density_summary(
        geometry=geometry,
        vertex_cache=vertex_cache,
        keys=keys,
        mu=mu,
    )
    result = SimpleNamespace(
        evaluator_evals=evaluator_evals,
        subdivisions=0,
        n_leaves=int(geometry.n_active),
        n_leaf_nodes=int(geometry.n_leaf_vertices),
        error_estimate_available=False,
    )
    info = _density_integration_info(result=result, spectral_cache=vertex_cache)
    return rho, error, info


def _root_mesh_fixed_filling_zero_temp(
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
):
    geometry, vertex_cache = _build_extension_runtime(h)
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        tol=float(_GEOM_TOL),
    )
    charge_integration_calls = 0
    charge_evaluator_evals = 0

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        nonlocal charge_integration_calls, charge_evaluator_evals
        (
            charge,
            derivative,
            derivative_exact,
            _owner_ids,
            _owner_charges,
            evaluator_evals,
        ) = integrator.evaluate_charge(float(mu), 0)
        charge_integration_calls += 1
        charge_evaluator_evals += int(evaluator_evals)
        resolved_derivative = (
            float(derivative)
            if bool(derivative_exact) and np.isfinite(derivative)
            else float("nan")
        )
        return float(charge), 0.0, resolved_derivative

    lower, upper = mu_bracket(h, 0.0)
    lower, upper = expand_mu_bracket(
        evaluate_charge,
        filling=filling,
        lower=lower,
        upper=upper,
    )
    mu, charge, _charge_error_unused, derivative, iteration = solve_mu(
        evaluate_charge,
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=charge_tol,
        mu_xtol=mu_xtol,
        max_mu_iterations=max_mu_iterations,
    )

    charge_kernel_evals = int(vertex_cache.n_kernel_evals)
    rho, error, density_evaluator_evals = _coarse_density_summary(
        geometry=geometry,
        vertex_cache=vertex_cache,
        keys=keys,
        mu=mu,
    )
    density_kernel_evals = int(vertex_cache.n_kernel_evals) - charge_kernel_evals

    charge_result = SimpleNamespace(
        mu=float(mu),
        charge=float(charge),
        charge_error=float("nan"),
        dcharge_dmu=float(derivative),
        root_iterations=int(iteration),
        charge_integration_calls=int(charge_integration_calls),
        evaluator_evals=int(charge_evaluator_evals),
        subdivisions=0,
        error_estimate_available=False,
    )
    density_result = SimpleNamespace(
        evaluator_evals=int(density_evaluator_evals),
        subdivisions=0,
        n_leaves=int(geometry.n_active),
        n_leaf_nodes=int(geometry.n_leaf_vertices),
        error_estimate_available=False,
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
    return rho, error, float(mu), info


def _build_charge_options(
    *,
    mu_guess: float,
    charge_tol: float,
    mu_xtol: float,
    max_mu_iterations: int,
    max_subdivisions: int | None,
):
    options = ChargeSolveOptions()
    options.mu_guess = float(mu_guess)
    options.charge_tol = float(charge_tol)
    options.mu_xtol = float(mu_xtol)
    options.max_mu_iterations = int(max_mu_iterations)
    options.max_subdivisions = _extension_subdivision_limit(max_subdivisions)
    options.bulk_theta = float(_BULK_THETA)
    return options


def _build_density_options(
    *,
    density_atol: float,
    density_rtol: float,
    max_subdivisions: int | None,
    consumed_subdivisions: int = 0,
):
    options = DensityIntegrateOptions()
    options.density_atol = float(density_atol)
    options.density_rtol = float(density_rtol)
    if max_subdivisions is None:
        options.max_subdivisions = -1
    else:
        options.max_subdivisions = max(
            int(max_subdivisions) - int(consumed_subdivisions), 0
        )
    options.bulk_theta = float(_BULK_THETA)
    return options


def _raise_normalized_runtime_error(exc: RuntimeError) -> None:
    if "Adaptive zero-temperature" in str(exc):
        raise ValueError(str(exc)) from exc
    raise exc


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
    max_subdivisions: int | None = None,
):
    """Evaluate the zero-temperature fixed-filling density matrix with the compiled backend."""

    if max_subdivisions == 0:
        return _root_mesh_fixed_filling_zero_temp(
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

    geometry, vertex_cache = _build_extension_runtime(h)
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        tol=float(_GEOM_TOL),
    )

    try:
        charge_result = integrator.solve_mu_and_refine(
            float(filling),
            _build_charge_options(
                mu_guess=mu_guess,
                charge_tol=charge_tol,
                mu_xtol=mu_xtol,
                max_mu_iterations=max_mu_iterations,
                max_subdivisions=max_subdivisions,
            ),
        )
    except RuntimeError as exc:
        _raise_normalized_runtime_error(exc)

    charge_kernel_evals = int(vertex_cache.n_kernel_evals)
    try:
        density_result = integrator.integrate_density(
            float(charge_result.mu),
            _build_density_options(
                density_atol=density_atol,
                density_rtol=density_rtol,
                max_subdivisions=max_subdivisions,
                consumed_subdivisions=int(charge_result.subdivisions),
            ),
        )
    except RuntimeError as exc:
        _raise_normalized_runtime_error(exc)

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
):
    """Evaluate the zero-temperature density matrix at an explicit chemical potential."""

    if max_subdivisions == 0:
        return _root_mesh_density_at_mu_zero_temp(h, mu=mu, keys=keys)

    geometry, vertex_cache = _build_extension_runtime(h)
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        tol=float(_GEOM_TOL),
    )
    try:
        density_result = integrator.integrate_density(
            float(mu),
            _build_density_options(
                density_atol=density_atol,
                density_rtol=density_rtol,
                max_subdivisions=max_subdivisions,
            ),
        )
    except RuntimeError as exc:
        _raise_normalized_runtime_error(exc)

    rho, error = _vector_to_density(
        np.asarray(density_result.estimate_array(), dtype=complex),
        np.asarray(density_result.error_vector_array(), dtype=float),
        int(vertex_cache.ndof),
        keys,
    )
    info = _density_integration_info(
        result=density_result, spectral_cache=vertex_cache
    )
    return rho, error, info
