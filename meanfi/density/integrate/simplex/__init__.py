from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import lineartetrahedron.backend as simplex_backend
from threadpoolctl import threadpool_limits

from meanfi.density.filling import FixedFillingSolve
from meanfi.density.filling import mu_bracket as build_mu_bracket
from meanfi.density.filling import solve_mu
from meanfi.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates
from meanfi.tb.ops import _tb_type

_ZERO_TEMP_EXT_AVAILABLE = bool(simplex_backend.NATIVE_AVAILABLE)
AdaptiveOptions = simplex_backend.AdaptiveOptions

_PREVIEW_DEPTH = 1
_MIN_REFINEMENT_BATCH_SIZE = 1
_MAX_REFINEMENT_BATCH_SIZE = 100


@dataclass(frozen=True)
class _PreparedDensityComponents:
    density_coordinates: DensityCoordinates
    key_array: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    key_indices: np.ndarray

    def values_and_errors_to_tb(
        self,
        values: np.ndarray,
        errors: np.ndarray,
    ) -> tuple[_tb_type, _tb_type]:
        return self.density_coordinates.values_and_errors_to_tb(values, errors)


def _require_native_backend() -> None:
    if (
        not bool(getattr(simplex_backend, "NATIVE_AVAILABLE", False))
        or getattr(simplex_backend, "IntegrationRuntime", None) is None
        or getattr(simplex_backend, "TightBindingModel", None) is None
    ):
        raise RuntimeError(
            "Zero-temperature integration requires the compiled "
            "lineartetrahedron._native extension"
        )


def _require_supported_dimension(h: _tb_type) -> None:
    ndim = len(next(iter(h)))
    if ndim not in (1, 2, 3):
        raise ValueError(
            "Adaptive simplex zero-temperature backend supports dimensions 1, 2, and 3"
        )


def _components_from_density_coordinates(
    density_coordinates: DensityCoordinates,
) -> list[tuple[int, int, tuple[int, ...]]]:
    return [
        (int(row), int(col), tuple(int(part) for part in key))
        for key, rows, cols, _value_slice in density_coordinates.iter_key_coordinates()
        for row, col in zip(rows, cols, strict=True)
    ]


def _resolve_density_components(
    h: _tb_type,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None,
) -> list[tuple[int, int, tuple[int, ...]]]:
    return _components_from_density_coordinates(
        _density_coordinates(h, keys=keys, density_coordinates=density_coordinates)
    )


def _density_coordinates(
    h: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None,
) -> DensityCoordinates:
    if density_coordinates is not None:
        return density_coordinates
    size = int(next(iter(h.values())).shape[0])
    return full_density_coordinates(keys, size=size)


def _prepare_density_components(
    h: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None,
) -> _PreparedDensityComponents:
    coords = _density_coordinates(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    key_index = {key: index for index, key in enumerate(coords.keys)}
    rows: list[int] = []
    cols: list[int] = []
    key_indices: list[int] = []
    for key, key_rows, key_cols, _value_slice in coords.iter_key_coordinates():
        rows.extend(int(row) for row in key_rows)
        cols.extend(int(col) for col in key_cols)
        key_indices.extend([key_index[key]] * len(key_rows))
    return _PreparedDensityComponents(
        density_coordinates=coords,
        key_array=np.ascontiguousarray(np.asarray(coords.keys, dtype=np.int64)),
        rows=np.ascontiguousarray(np.asarray(rows, dtype=np.int64)),
        cols=np.ascontiguousarray(np.asarray(cols, dtype=np.int64)),
        key_indices=np.ascontiguousarray(np.asarray(key_indices, dtype=np.int64)),
    )


def _max_refinements(max_subdivisions: int | None) -> int:
    return -1 if max_subdivisions is None else int(max_subdivisions)


def _native_thread_context(num_threads: int | None):
    if num_threads is None:
        return nullcontext()
    return threadpool_limits(limits=int(num_threads), user_api="openmp")


def _adaptive_options(
    *,
    target_error: float,
    max_refinements: int,
    num_threads: int | None,
):
    del num_threads
    return AdaptiveOptions(
        float(target_error),
        max_refinements=max_refinements,
        preview_depth=_PREVIEW_DEPTH,
        min_refinement_batch_size=_MIN_REFINEMENT_BATCH_SIZE,
        max_refinement_batch_size=_MAX_REFINEMENT_BATCH_SIZE,
    )


def _call_native(method, *args, num_threads: int | None):
    if num_threads is not None:
        try:
            return method(*args, int(num_threads))
        except TypeError:
            pass
    with _native_thread_context(num_threads):
        return method(*args)


def _integrate_charge(
    runtime,
    *,
    mu: float,
    charge_tol: float,
    max_refinements: int,
    num_threads: int | None,
):
    return _call_native(
        runtime.integrate_charge,
        mu,
        _adaptive_options(
            target_error=charge_tol,
            max_refinements=max_refinements,
            num_threads=num_threads,
        ),
        True,
        True,
        0.0,
        0.0,
        num_threads=num_threads,
    )


def _evaluate_charge(
    runtime,
    *,
    mu: float,
    charge_tol: float,
    num_threads: int | None,
):
    return _call_native(
        runtime.integrate_charge,
        mu,
        _adaptive_options(
            target_error=charge_tol,
            max_refinements=0,
            num_threads=num_threads,
        ),
        False,
        False,
        0.0,
        0.0,
        num_threads=num_threads,
    )


def _integrate_density(
    runtime,
    prepared: _PreparedDensityComponents,
    *,
    mu: float,
    density_atol: float,
    max_refinements: int,
    num_threads: int | None,
):
    return _call_native(
        runtime.integrate_density,
        mu,
        _adaptive_options(
            target_error=density_atol,
            max_refinements=max_refinements,
            num_threads=num_threads,
        ),
        prepared.key_array,
        prepared.rows,
        prepared.cols,
        prepared.key_indices,
        True,
        num_threads=num_threads,
    )


def _runtime_and_components(
    h: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None,
):
    _require_native_backend()
    _require_supported_dimension(h)
    prepared = _prepare_density_components(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    model = simplex_backend._tb_to_tight_binding_model(h)
    runtime = simplex_backend.IntegrationRuntime(model)
    return runtime, prepared


def _raise_if_not_converged(result, message: str) -> None:
    if not bool(result.converged):
        raise RuntimeError(message)


def _density_info(
    result,
    runtime,
    *,
    num_threads: int | None,
) -> DensityIntegrationInfo:
    work = int(result.work)
    return DensityIntegrationInfo(
        n_kernel_evals=work,
        unique_evals=work,
        n_evaluator_evals=work,
        n_cached_nodes=int(runtime.n_cached_nodes),
        n_leaves=int(result.n_active_simplices),
        n_leaf_nodes=int(result.n_active_vertices),
        subdivisions=int(result.refinements),
        error_estimate_available=bool(result.converged),
        num_threads=num_threads,
    )


def density_matrix_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
    density_atol: float,
    density_rtol: float,
    max_subdivisions: int | None = None,
    num_threads: int | None = None,
):
    del density_rtol
    runtime, prepared = _runtime_and_components(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    result = _integrate_density(
        runtime,
        prepared,
        mu=float(mu),
        density_atol=float(density_atol),
        max_refinements=_max_refinements(max_subdivisions),
        num_threads=num_threads,
    )
    _raise_if_not_converged(
        result,
        "Adaptive simplex loop did not converge while evaluating density",
    )
    density_matrix, density_matrix_error = prepared.values_and_errors_to_tb(
        result.estimate_array(),
        result.error_vector_array(),
    )
    return (
        density_matrix,
        density_matrix_error,
        _density_info(result, runtime, num_threads=num_threads),
    )


def _fixed_filling_info(
    *,
    root,
    charge_integration_calls: int,
    charge_work: int,
    charge_refinements: int,
    density_info: DensityIntegrationInfo,
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
    num_threads: int | None,
) -> FixedFillingInfo:
    return FixedFillingInfo(
        mu=float(root.mu),
        charge=float(root.charge),
        charge_error=float(root.charge_error),
        dcharge_dmu=0.0 if root.derivative is None else float(root.derivative),
        charge_evaluations=int(root.charge_evaluations),
        charge_integration_calls=int(charge_integration_calls),
        density_integration_calls=1,
        charge_n_kernel_evals=int(charge_work),
        density_n_kernel_evals=int(density_info.n_kernel_evals),
        n_kernel_evals=int(charge_work + density_info.n_kernel_evals),
        unique_evals=int(charge_work + density_info.unique_evals),
        charge_n_evaluator_evals=int(charge_work),
        density_n_evaluator_evals=int(density_info.n_evaluator_evals),
        n_evaluator_evals=int(charge_work + density_info.n_evaluator_evals),
        n_cached_nodes=int(density_info.n_cached_nodes),
        n_leaves=int(density_info.n_leaves),
        n_leaf_nodes=int(density_info.n_leaf_nodes),
        subdivisions=int(charge_refinements + density_info.subdivisions),
        charge_integral_atol=float(charge_tol),
        density_atol=float(density_atol),
        density_rtol=float(density_rtol),
        error_estimate_available=bool(density_info.error_estimate_available),
        num_threads=num_threads,
    )


def density_matrix_zero_temp(
    h: _tb_type,
    *,
    filling: float,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
    charge_tol: float,
    filling_tol: float,
    density_atol: float,
    density_rtol: float,
    mu_guess: float,
    mu_xtol: float,
    max_charge_evaluations: int | None,
    max_subdivisions: int | None = None,
    num_threads: int | None = None,
):
    runtime, prepared = _runtime_and_components(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    max_refinements = _max_refinements(max_subdivisions)
    charge_integration_calls = 0
    charge_work = 0
    charge_refinements = 0
    charge_evaluations = 0
    current_mu_guess = float(mu_guess)

    while True:
        active_simplices = int(runtime.n_active_simplices)

        def evaluate_charge(candidate_mu: float) -> tuple[float, float, float | None]:
            nonlocal charge_integration_calls, charge_work
            result = _evaluate_charge(
                runtime,
                mu=float(candidate_mu),
                charge_tol=float(charge_tol),
                num_threads=num_threads,
            )
            if int(result.n_active_simplices) != active_simplices:
                raise RuntimeError("Charge evaluation changed the active simplex mesh")
            charge_integration_calls += 1
            charge_work += int(result.work)
            return (
                float(result.charge),
                float(result.charge_error),
                float(result.dcharge_dmu),
            )

        remaining_charge_evaluations = (
            None
            if max_charge_evaluations is None
            else max_charge_evaluations - charge_evaluations
        )
        if (
            remaining_charge_evaluations is not None
            and remaining_charge_evaluations <= 0
        ):
            raise RuntimeError(
                "Chemical-potential solve failed: maximum charge-evaluation budget "
                "reached before satisfying the filling tolerance"
            )

        root = solve_mu(
            evaluate_charge=evaluate_charge,
            initial_bracket=lambda: build_mu_bracket(h, 0.0),
            filling=float(filling),
            mu_guess=current_mu_guess,
            filling_tol=float(filling_tol),
            mu_tol=float(mu_xtol),
            max_charge_evaluations=remaining_charge_evaluations,
            use_derivative=True,
        )
        charge_evaluations += int(root.charge_evaluations)
        current_mu_guess = float(root.mu)

        remaining_refinements = (
            max_refinements
            if max_refinements < 0
            else max_refinements - charge_refinements
        )
        result = _integrate_charge(
            runtime,
            mu=float(root.mu),
            charge_tol=float(charge_tol),
            max_refinements=remaining_refinements,
            num_threads=num_threads,
        )
        charge_integration_calls += 1
        charge_work += int(result.work)
        charge_refinements += int(result.refinements)
        _raise_if_not_converged(
            result,
            "Adaptive simplex loop did not converge while solving for the chemical potential",
        )

        residual = float(result.charge) - float(filling)
        if abs(residual) <= float(filling_tol) and float(result.charge_error) <= float(
            charge_tol
        ):
            root = FixedFillingSolve(
                mu=float(root.mu),
                charge=float(result.charge),
                charge_error=float(result.charge_error),
                residual=residual,
                derivative=float(result.dcharge_dmu),
                charge_evaluations=charge_evaluations,
            )
            break

    root = FixedFillingSolve(
        mu=float(root.mu),
        charge=float(root.charge),
        charge_error=float(root.charge_error),
        residual=float(root.residual),
        derivative=root.derivative,
        charge_evaluations=charge_evaluations,
    )
    density_result = _integrate_density(
        runtime,
        prepared,
        mu=float(root.mu),
        density_atol=float(density_atol),
        max_refinements=max_refinements,
        num_threads=num_threads,
    )
    _raise_if_not_converged(
        density_result,
        "Adaptive simplex loop did not converge while evaluating density",
    )
    density_matrix, density_matrix_error = prepared.values_and_errors_to_tb(
        density_result.estimate_array(),
        density_result.error_vector_array(),
    )
    density_info = _density_info(density_result, runtime, num_threads=num_threads)
    return (
        density_matrix,
        density_matrix_error,
        float(root.mu),
        _fixed_filling_info(
            root=root,
            charge_integration_calls=charge_integration_calls,
            charge_work=charge_work,
            charge_refinements=charge_refinements,
            density_info=density_info,
            charge_tol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            num_threads=num_threads,
        ),
    )


__all__ = [
    "_ZERO_TEMP_EXT_AVAILABLE",
    "density_matrix_at_mu_zero_temp",
    "density_matrix_zero_temp",
]
