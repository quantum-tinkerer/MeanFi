from __future__ import annotations

from contextlib import nullcontext

import numpy as np
from fermisimplex import SpectralMesh
from threadpoolctl import threadpool_limits

from meanfi.density.filling import FixedFillingSolve
from meanfi.density.filling import mu_bracket as build_mu_bracket
from meanfi.density.filling import solve_mu
from meanfi.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates
from meanfi.tb.ops import _tb_type, to_dense


_ZERO_TEMP_EXT_AVAILABLE = True
_CHARGE_ERROR_DEPTH = 2
_DENSITY_PREVIEW_DEPTH = 1
_MIN_REFINEMENT_BATCH_SIZE = 1
_MAX_REFINEMENT_BATCH_SIZE = 100


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


def _spectral_mesh(h: _tb_type) -> SpectralMesh:
    dense_hamiltonian = {
        key: np.asarray(to_dense(matrix), dtype=np.complex128)
        for key, matrix in h.items()
    }
    return SpectralMesh(dense_hamiltonian)


def _native_thread_context(num_threads: int | None):
    if num_threads is None:
        return nullcontext()
    return threadpool_limits(limits=int(num_threads), user_api="openmp")


def _integrate_charge(
    mesh: SpectralMesh,
    *,
    mu: float,
    charge_tol: float,
    max_refinements: int | None,
    num_threads: int | None,
):
    with _native_thread_context(num_threads):
        return mesh.integrate_charge(
            mu=float(mu),
            target_error=float(charge_tol),
            max_refinements=max_refinements,
            error_depth=_CHARGE_ERROR_DEPTH,
            min_refinement_batch_size=_MIN_REFINEMENT_BATCH_SIZE,
            max_refinement_batch_size=_MAX_REFINEMENT_BATCH_SIZE,
        )


def _evaluate_charge(
    mesh: SpectralMesh,
    *,
    mu: float,
    num_threads: int | None,
):
    cached_vertices = int(mesh.cached_vertices)
    with _native_thread_context(num_threads):
        result = mesh.estimate_charge_on_current_mesh(mu=float(mu))
    return result, int(mesh.cached_vertices) - cached_vertices


def _occupied_band_energy(
    mesh: SpectralMesh,
    *,
    mu: float,
) -> float | None:
    if not hasattr(mesh, "occupied_weights"):
        return None
    weights = np.asarray(mesh.occupied_weights(float(mu)))
    return float(np.sum(weights * np.asarray(mesh.eigenvalues)))


def _integrate_density(
    mesh: SpectralMesh,
    density_coordinates: DensityCoordinates,
    *,
    mu: float,
    density_atol: float,
    max_refinements: int | None,
    num_threads: int | None,
):
    key_indices = {key: index for index, key in enumerate(density_coordinates.keys)}
    components = np.asarray(
        [(key_indices[key], row, col) for key, row, col in density_coordinates.entries],
        dtype=np.int64,
    ).reshape((-1, 3))
    with _native_thread_context(num_threads):
        return mesh.integrate_density_components(
            mu=float(mu),
            lattice_vectors=density_coordinates.keys,
            components=components,
            target_error=float(density_atol),
            max_refinements=max_refinements,
            preview_depth=_DENSITY_PREVIEW_DEPTH,
            min_refinement_batch_size=_MIN_REFINEMENT_BATCH_SIZE,
            max_refinement_batch_size=_MAX_REFINEMENT_BATCH_SIZE,
        )


def _density_result_to_tb(
    result,
    density_coordinates: DensityCoordinates,
) -> tuple[_tb_type, _tb_type]:
    values = np.asarray(result.values)
    errors = np.full(
        density_coordinates.value_count,
        float(result.stopping_error),
        dtype=float,
    )
    return density_coordinates.values_and_errors_to_tb(values, errors)


def _empty_density_result(
    mesh: SpectralMesh,
    density_coordinates: DensityCoordinates,
    *,
    num_threads: int | None,
) -> tuple[_tb_type, _tb_type, DensityIntegrationInfo]:
    density_matrix, density_matrix_error = density_coordinates.values_and_errors_to_tb(
        np.empty(0, dtype=complex),
        np.empty(0, dtype=float),
    )
    return (
        density_matrix,
        density_matrix_error,
        DensityIntegrationInfo(
            n_kernel_evals=0,
            unique_evals=0,
            n_evaluator_evals=0,
            n_cached_nodes=int(mesh.cached_vertices),
            n_leaves=int(mesh.active_simplices),
            n_leaf_nodes=int(mesh.active_vertices),
            subdivisions=0,
            error_estimate_available=True,
            num_threads=num_threads,
        ),
    )


def _raise_if_not_converged(result, message: str) -> None:
    if not bool(result.stats.target_reached):
        raise RuntimeError(message)


def _density_info(
    result,
    *,
    num_threads: int | None,
) -> DensityIntegrationInfo:
    stats = result.stats
    evaluations = int(stats.evaluations)
    return DensityIntegrationInfo(
        n_kernel_evals=evaluations,
        unique_evals=evaluations,
        n_evaluator_evals=evaluations,
        n_cached_nodes=int(stats.cached_vertices),
        n_leaves=int(stats.active_simplices),
        n_leaf_nodes=int(stats.active_vertices),
        subdivisions=int(stats.refinements),
        error_estimate_available=bool(stats.target_reached),
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
    coordinates = _density_coordinates(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    mesh = _spectral_mesh(h)
    if coordinates.value_count == 0:
        density_matrix, density_matrix_error, density_info = _empty_density_result(
            mesh, coordinates, num_threads=num_threads
        )
        return density_matrix, density_matrix_error, density_info

    result = _integrate_density(
        mesh,
        coordinates,
        mu=float(mu),
        density_atol=float(density_atol),
        max_refinements=max_subdivisions,
        num_threads=num_threads,
    )
    _raise_if_not_converged(
        result,
        "Adaptive simplex loop did not converge while evaluating density",
    )
    density_matrix, density_matrix_error = _density_result_to_tb(
        result,
        coordinates,
    )
    return (
        density_matrix,
        density_matrix_error,
        _density_info(result, num_threads=num_threads),
    )


def _fixed_filling_info(
    *,
    root,
    charge_integration_calls: int,
    density_integration_calls: int,
    charge_work: int,
    charge_refinements: int,
    density_info: DensityIntegrationInfo,
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
    num_threads: int | None,
    band_energy: float | None = None,
) -> FixedFillingInfo:
    return FixedFillingInfo(
        mu=float(root.mu),
        charge=float(root.charge),
        charge_error=float(root.charge_error),
        dcharge_dmu=0.0 if root.derivative is None else float(root.derivative),
        charge_evaluations=int(root.charge_evaluations),
        charge_integration_calls=int(charge_integration_calls),
        density_integration_calls=int(density_integration_calls),
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
        band_energy=band_energy,
        band_energy_integration_calls=int(band_energy is not None),
        band_energy_n_kernel_evals=0,
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
    include_band_energy: bool = False,
):
    coordinates = _density_coordinates(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    mesh = _spectral_mesh(h)
    charge_integration_calls = 0
    charge_work = 0
    charge_refinements = 0
    charge_evaluations = 0
    current_mu_guess = float(mu_guess)

    while True:

        def evaluate_charge(candidate_mu: float) -> tuple[float, float, float | None]:
            nonlocal charge_integration_calls, charge_work
            result, evaluations = _evaluate_charge(
                mesh,
                mu=float(candidate_mu),
                num_threads=num_threads,
            )
            charge_integration_calls += 1
            charge_work += evaluations
            return float(result.value), 0.0, float(result.dcharge_dmu)

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
            filling_tol=min(float(filling_tol), float(charge_tol)),
            mu_tol=float(mu_xtol),
            max_charge_evaluations=remaining_charge_evaluations,
            use_derivative=True,
        )
        charge_evaluations += int(root.charge_evaluations)
        current_mu_guess = float(root.mu)

        remaining_refinements = (
            None if max_subdivisions is None else max_subdivisions - charge_refinements
        )
        result = _integrate_charge(
            mesh,
            mu=float(root.mu),
            charge_tol=float(charge_tol),
            max_refinements=remaining_refinements,
            num_threads=num_threads,
        )
        charge_integration_calls += 1
        charge_work += int(result.stats.evaluations)
        charge_work += int(result.error_stats.hamiltonian_evaluations)
        charge_refinements += int(result.stats.refinements)
        _raise_if_not_converged(
            result,
            "Adaptive simplex loop did not converge while solving for the chemical potential",
        )

        residual = float(result.value) - float(filling)
        if abs(residual) <= float(filling_tol) and float(
            result.stopping_error
        ) <= float(charge_tol):
            root = FixedFillingSolve(
                mu=float(root.mu),
                charge=float(result.value),
                charge_error=float(result.stopping_error),
                residual=residual,
                derivative=float(result.dcharge_dmu),
                charge_evaluations=charge_evaluations,
            )
            break

    if coordinates.value_count == 0:
        density_matrix, density_matrix_error, density_info = _empty_density_result(
            mesh, coordinates, num_threads=num_threads
        )
        band_energy = (
            _occupied_band_energy(mesh, mu=float(root.mu))
            if include_band_energy
            else None
        )
        return (
            density_matrix,
            density_matrix_error,
            float(root.mu),
            _fixed_filling_info(
                root=root,
                charge_integration_calls=charge_integration_calls,
                density_integration_calls=0,
                charge_work=charge_work,
                charge_refinements=charge_refinements,
                density_info=density_info,
                charge_tol=charge_tol,
                density_atol=density_atol,
                density_rtol=density_rtol,
                num_threads=num_threads,
                band_energy=band_energy,
            ),
        )

    density_result = _integrate_density(
        mesh,
        coordinates,
        mu=float(root.mu),
        density_atol=float(density_atol),
        max_refinements=max_subdivisions,
        num_threads=num_threads,
    )
    _raise_if_not_converged(
        density_result,
        "Adaptive simplex loop did not converge while evaluating density",
    )
    density_matrix, density_matrix_error = _density_result_to_tb(
        density_result,
        coordinates,
    )
    density_info = _density_info(density_result, num_threads=num_threads)
    band_energy = (
        _occupied_band_energy(mesh, mu=float(root.mu)) if include_band_energy else None
    )
    return (
        density_matrix,
        density_matrix_error,
        float(root.mu),
        _fixed_filling_info(
            root=root,
            charge_integration_calls=charge_integration_calls,
            density_integration_calls=1,
            charge_work=charge_work,
            charge_refinements=charge_refinements,
            density_info=density_info,
            charge_tol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            num_threads=num_threads,
            band_energy=band_energy,
        ),
    )


__all__ = [
    "_ZERO_TEMP_EXT_AVAILABLE",
    "density_matrix_at_mu_zero_temp",
    "density_matrix_zero_temp",
]
