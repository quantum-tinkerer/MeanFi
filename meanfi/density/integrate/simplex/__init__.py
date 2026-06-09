from __future__ import annotations

from lineartetrahedron import NATIVE_AVAILABLE as _ZERO_TEMP_EXT_AVAILABLE
from lineartetrahedron import (
    AdaptiveOptions,
    build_runtime,
    full_density_components,
    prepare_density_components,
)

from meanfi.density.filling import FixedFillingSolve
from meanfi.density.filling import mu_bracket as build_mu_bracket
from meanfi.density.filling import solve_mu
from meanfi.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import _tb_type

_PREVIEW_DEPTH = 3
_MIN_REFINEMENT_BATCH_SIZE = 1
_MAX_REFINEMENT_BATCH_SIZE = 100
_ROOT_SOLVE_CHARGE_ERROR_TOL = 1e300


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
    if density_coordinates is not None:
        return _components_from_density_coordinates(density_coordinates)
    size = int(next(iter(h.values())).shape[0])
    return full_density_components(keys, size=size)


def _max_refinements(max_subdivisions: int | None) -> int:
    return -1 if max_subdivisions is None else int(max_subdivisions)


def _native_threading_error() -> RuntimeError:
    return RuntimeError(
        "AdaptiveSimplex(num_threads=...) is not supported by this "
        "lineartetrahedron backend"
    )


def _adaptive_options(
    *,
    target_error: float,
    max_refinements: int,
    num_threads: int | None,
):
    if num_threads is not None:
        raise _native_threading_error()
    return AdaptiveOptions(
        float(target_error),
        max_refinements=max_refinements,
        preview_depth=_PREVIEW_DEPTH,
        min_refinement_batch_size=_MIN_REFINEMENT_BATCH_SIZE,
        max_refinement_batch_size=_MAX_REFINEMENT_BATCH_SIZE,
    )


def _integrate_charge(
    runtime,
    *,
    mu: float,
    charge_tol: float,
    max_refinements: int,
    num_threads: int | None,
):
    return runtime.integrate_charge(
        mu,
        _adaptive_options(
            target_error=charge_tol,
            max_refinements=max_refinements,
            num_threads=num_threads,
        ),
    )


def _evaluate_charge(
    runtime,
    *,
    mu: float,
    charge_tol: float,
    num_threads: int | None,
):
    return runtime.evaluate_charge(
        mu,
        _adaptive_options(
            target_error=charge_tol,
            max_refinements=0,
            num_threads=num_threads,
        ),
    )


def _integrate_density(
    runtime,
    *,
    mu: float,
    density_atol: float,
    max_refinements: int,
    num_threads: int | None,
):
    return runtime.integrate_density(
        mu,
        _adaptive_options(
            target_error=density_atol,
            max_refinements=max_refinements,
            num_threads=num_threads,
        ),
    )


def _runtime_and_components(
    h: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None,
):
    prepared = prepare_density_components(
        h,
        keys,
        _resolve_density_components(h, keys, density_coordinates),
    )
    runtime = build_runtime(
        h,
        keys=list(prepared.keys),
        component_rows=prepared.rows,
        component_cols=prepared.cols,
        component_key_indices=prepared.key_indices,
    )
    return runtime, prepared


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
        mu=float(mu),
        density_atol=float(density_atol),
        max_refinements=_max_refinements(max_subdivisions),
        num_threads=num_threads,
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
            charge_error_tol=_ROOT_SOLVE_CHARGE_ERROR_TOL,
            use_derivative=True,
        )
        charge_evaluations += int(root.charge_evaluations)
        current_mu_guess = float(root.mu)

        if float(root.charge_error) <= float(charge_tol):
            break

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
        mu=float(root.mu),
        density_atol=float(density_atol),
        max_refinements=max_refinements,
        num_threads=num_threads,
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
