from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from math import comb

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


def _spectral_mesh(
    h: _tb_type,
    *,
    nk: int | None = None,
    max_points: int | None = None,
) -> SpectralMesh:
    # FermiSimplex's dyadic root mesh includes both faces of the unit cell.
    # Its native construction gives (2**level + 1)**dimension distinct nodes.
    dimension = len(next(iter(h)))
    level = 1 if nk is None else 0
    while nk is not None and (2**level + 1) ** dimension < nk:
        level += 1
    nodes = (2**level + 1) ** dimension
    if max_points is not None and nodes > max_points:
        raise RuntimeError(
            f"AdaptiveSimplex mesh requires {nodes} nodes for nk={nk}, "
            f"exceeding max_points={max_points}; increase max_points or reduce nk"
        )
    dense_hamiltonian = {
        key: np.asarray(to_dense(matrix), dtype=np.complex128)
        for key, matrix in h.items()
    }
    return SpectralMesh(dense_hamiltonian, root_level=level)


def _bounded_refinements(
    mesh: SpectralMesh,
    max_refinements: int | None,
    max_points: int | None,
    *,
    preview_depth: int,
) -> int | None:
    """Conservatively bound native retained spectra, including density previews.

    Each native refinement splits one simplex into 2**dimension children.
    Reserving their vertices before the call avoids an unbounded native cache
    without implementing another refinement engine. Shared vertices can make
    this reserve larger than the actual allocation.
    """
    if max_points is None:
        return max_refinements
    dimension = int(mesh.ndim)
    nodes_per_simplex = comb(dimension + 2**preview_depth, dimension)
    initial = max(int(mesh.cached_vertices), int(mesh.active_vertices))
    initial += int(mesh.active_simplices) * (nodes_per_simplex - dimension - 1)
    if initial > max_points:
        raise RuntimeError(
            f"AdaptiveSimplex needs a reserve of {initial} cached/preview nodes, "
            f"exceeding max_points={max_points}; increase max_points or loosen tolerances"
        )
    per_refinement = 2**dimension * nodes_per_simplex
    available = (max_points - initial) // per_refinement
    return available if max_refinements is None else min(max_refinements, available)


def _native_thread_context(num_threads: int | None):
    if num_threads is None:
        return nullcontext()
    return threadpool_limits(limits=int(num_threads), user_api="openmp")


@contextmanager
def _integration_context(num_threads: int | None):
    try:
        with _native_thread_context(num_threads):
            yield
    except RuntimeError as error:
        if "did not converge" not in str(error):
            raise
        raise RuntimeError(
            f"{error}; increase max_points/max_refinements or loosen integration tolerances"
        ) from error


def _integrate_charge(
    mesh: SpectralMesh,
    *,
    mu: float,
    charge_tol: float,
    max_refinements: int | None,
    num_threads: int | None,
    max_points: int | None = None,
):
    max_refinements = _bounded_refinements(
        mesh, max_refinements, max_points, preview_depth=0
    )
    with _integration_context(num_threads):
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
    prescribed: bool = False,
    max_points: int | None = None,
):
    preview_depth = 0 if prescribed else _DENSITY_PREVIEW_DEPTH
    max_refinements = _bounded_refinements(
        mesh, max_refinements, max_points, preview_depth=preview_depth
    )
    key_indices = {key: index for index, key in enumerate(density_coordinates.keys)}
    components = np.asarray(
        [(key_indices[key], row, col) for key, row, col in density_coordinates.entries],
        dtype=np.int64,
    ).reshape((-1, 3))
    with _integration_context(num_threads):
        return mesh.integrate_density_components(
            mu=float(mu),
            lattice_vectors=density_coordinates.keys,
            components=components,
            target_error=float(density_atol),
            max_refinements=max_refinements,
            preview_depth=preview_depth,
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
    nk: int | None = None,
) -> tuple[_tb_type, _tb_type | None, DensityIntegrationInfo]:
    density_matrix, density_matrix_error = density_coordinates.values_and_errors_to_tb(
        np.empty(0, dtype=complex),
        np.empty(0, dtype=float),
    )
    return (
        density_matrix,
        density_matrix_error if nk is None else None,
        DensityIntegrationInfo(
            n_kernel_evals=0,
            unique_evals=0,
            n_evaluator_evals=0,
            n_cached_nodes=int(mesh.cached_vertices),
            n_leaves=int(mesh.active_simplices),
            n_leaf_nodes=int(mesh.active_vertices),
            subdivisions=0,
            error_estimate_available=nk is None,
            num_threads=num_threads,
            requested_nk=nk,
            n_kpoints=int(mesh.active_vertices),
            n_diagonalizations=0,
        ),
    )


def _raise_if_not_converged(result, message: str) -> None:
    if not bool(result.stats.target_reached):
        raise RuntimeError(message)


def _density_info(
    result,
    *,
    num_threads: int | None,
    nk: int | None = None,
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
        error_estimate_available=nk is None and bool(stats.target_reached),
        num_threads=num_threads,
        requested_nk=nk,
        n_kpoints=int(stats.active_vertices),
        n_diagonalizations=evaluations,
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
    nk: int | None = None,
    max_points: int | None = None,
    charge_tol: float | None = None,
):
    del density_rtol
    coordinates = _density_coordinates(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    mesh = _spectral_mesh(h, nk=nk, max_points=max_points)
    work = diagonalizations = refinements = 0
    charge_error = None
    while True:
        remaining = None if max_subdivisions is None else max_subdivisions - refinements
        if coordinates.value_count == 0:
            density_matrix, density_matrix_error, density_info = _empty_density_result(
                mesh, coordinates, num_threads=num_threads, nk=nk
            )
        else:
            result = _integrate_density(
                mesh,
                coordinates,
                mu=float(mu),
                density_atol=0.0 if nk is not None else float(density_atol),
                max_refinements=remaining,
                num_threads=num_threads,
                prescribed=nk is not None,
                max_points=max_points,
            )
            _raise_if_not_converged(
                result,
                "Adaptive simplex loop did not converge while evaluating density",
            )
            density_matrix, density_matrix_error = _density_result_to_tb(
                result, coordinates
            )
            density_info = _density_info(result, num_threads=num_threads, nk=nk)
            work += int(result.stats.evaluations)
            diagonalizations += int(result.stats.evaluations)
            refinements += int(result.stats.refinements)

        # Charge is independent of the requested density layout. Prescribed
        # meshes are evaluated directly; adaptive calls must meet both targets.
        if nk is not None:
            charge, evaluations = _evaluate_charge(mesh, mu=mu, num_threads=num_threads)
            work += evaluations
            diagonalizations += evaluations
            break
        remaining = None if max_subdivisions is None else max_subdivisions - refinements
        charge = _integrate_charge(
            mesh,
            mu=float(mu),
            charge_tol=float(density_atol if charge_tol is None else charge_tol),
            max_refinements=remaining,
            num_threads=num_threads,
            max_points=max_points,
        )
        _raise_if_not_converged(
            charge, "Adaptive simplex loop did not converge while evaluating charge"
        )
        work += int(charge.stats.evaluations) + int(
            charge.error_stats.hamiltonian_evaluations
        )
        diagonalizations += int(charge.stats.evaluations) + sum(
            int(getattr(charge.error_stats, name, 0))
            for name in (
                "full_eigensystems",
                "reduced_eigensystems",
                "norm_eigensystems",
            )
        )
        refinements += int(charge.stats.refinements)
        charge_error = float(charge.stopping_error)
        if coordinates.value_count == 0 or int(charge.stats.refinements) == 0:
            break
        # Charge refinement changed the mesh: evaluate density again before
        # accepting a pair of estimates on the same final native mesh.

    density_info = replace(
        density_info,
        charge=float(charge.value),
        charge_error=charge_error,
        n_kernel_evals=work,
        unique_evals=work,
        n_evaluator_evals=work,
        n_diagonalizations=diagonalizations,
        subdivisions=refinements,
        n_cached_nodes=int(mesh.cached_vertices),
        n_leaves=int(mesh.active_simplices),
        n_leaf_nodes=int(mesh.active_vertices),
        n_kpoints=int(mesh.active_vertices),
    )
    return density_matrix, density_matrix_error if nk is None else None, density_info


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
    nk: int | None = None,
    charge_diagonalizations: int = 0,
) -> FixedFillingInfo:
    return FixedFillingInfo(
        mu=float(root.mu),
        charge=float(root.charge),
        charge_error=float(root.charge_error) if nk is None else None,
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
        requested_nk=nk,
        n_kpoints=density_info.n_leaf_nodes,
        n_diagonalizations=charge_diagonalizations + density_info.n_kernel_evals,
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
    nk: int | None = None,
    max_points: int | None = None,
):
    coordinates = _density_coordinates(
        h,
        keys=keys,
        density_coordinates=density_coordinates,
    )
    mesh = _spectral_mesh(h, nk=nk, max_points=max_points)
    charge_integration_calls = 0
    charge_work = 0
    charge_diagonalizations = 0
    charge_refinements = 0
    charge_evaluations = 0
    density_integration_calls = 0
    density_work = 0
    density_refinements = 0
    current_mu_guess = float(mu_guess)

    def remaining_refinements():
        return (
            None
            if max_subdivisions is None
            else max_subdivisions - charge_refinements - density_refinements
        )

    def evaluate_charge(candidate_mu: float) -> tuple[float, float, float | None]:
        nonlocal charge_integration_calls, charge_work, charge_diagonalizations
        result, evaluations = _evaluate_charge(
            mesh, mu=float(candidate_mu), num_threads=num_threads
        )
        charge_integration_calls += 1
        charge_work += evaluations
        charge_diagonalizations += evaluations
        # This root searches a frozen native discretization. Integration error
        # is estimated separately after root finding, never assumed to be zero.
        return float(result.value), 0.0, float(result.dcharge_dmu)

    def integrate_charge(candidate_mu: float):
        nonlocal charge_integration_calls, charge_work
        nonlocal charge_diagonalizations, charge_refinements
        result = _integrate_charge(
            mesh,
            mu=candidate_mu,
            charge_tol=float(charge_tol),
            max_refinements=remaining_refinements(),
            num_threads=num_threads,
            max_points=max_points,
        )
        charge_integration_calls += 1
        charge_work += int(result.stats.evaluations)
        charge_work += int(result.error_stats.hamiltonian_evaluations)
        charge_diagonalizations += int(result.stats.evaluations) + sum(
            int(getattr(result.error_stats, name, 0))
            for name in (
                "full_eigensystems",
                "reduced_eigensystems",
                "norm_eigensystems",
            )
        )
        charge_refinements += int(result.stats.refinements)
        _raise_if_not_converged(
            result,
            "Adaptive simplex loop did not converge while solving for the chemical potential",
        )
        return result

    while True:
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
            filling_tol=(
                float(filling_tol)
                if nk is not None
                else min(float(filling_tol), float(charge_tol))
            ),
            mu_tol=float(mu_xtol),
            max_charge_evaluations=remaining_charge_evaluations,
            use_derivative=True,
        )
        charge_evaluations += int(root.charge_evaluations)
        current_mu_guess = float(root.mu)
        if nk is None:
            charge_result = integrate_charge(float(root.mu))
            if abs(float(charge_result.value) - filling) > filling_tol:
                continue

        if coordinates.value_count == 0:
            density_matrix, density_matrix_error, density_info = _empty_density_result(
                mesh, coordinates, num_threads=num_threads, nk=nk
            )
        else:
            density_result = _integrate_density(
                mesh,
                coordinates,
                mu=float(root.mu),
                density_atol=0.0 if nk is not None else float(density_atol),
                max_refinements=remaining_refinements(),
                num_threads=num_threads,
                prescribed=nk is not None,
                max_points=max_points,
            )
            _raise_if_not_converged(
                density_result,
                "Adaptive simplex loop did not converge while evaluating density",
            )
            density_integration_calls += 1
            density_work += int(density_result.stats.evaluations)
            density_refinements += int(density_result.stats.refinements)
            density_matrix, density_matrix_error = _density_result_to_tb(
                density_result, coordinates
            )
            density_info = _density_info(density_result, num_threads=num_threads, nk=nk)

            if nk is None and int(density_result.stats.refinements) > 0:
                # Density refinement can change the charge at the solved mu.
                # Recheck on that mesh; repeat the solve if either integral
                # still needs a different mesh or chemical potential.
                charge_result = integrate_charge(float(root.mu))
                if (
                    int(charge_result.stats.refinements) > 0
                    or abs(float(charge_result.value) - filling) > filling_tol
                ):
                    continue

        if nk is None:
            root = FixedFillingSolve(
                mu=float(root.mu),
                charge=float(charge_result.value),
                charge_error=float(charge_result.stopping_error),
                residual=float(charge_result.value) - filling,
                derivative=float(charge_result.dcharge_dmu),
                charge_evaluations=charge_evaluations,
            )
        break

    density_info = replace(
        density_info,
        n_kernel_evals=density_work,
        unique_evals=density_work,
        n_evaluator_evals=density_work,
        n_diagonalizations=density_work,
        subdivisions=density_refinements,
    )
    band_energy = (
        _occupied_band_energy(mesh, mu=float(root.mu)) if include_band_energy else None
    )
    return (
        density_matrix,
        density_matrix_error if nk is None else None,
        float(root.mu),
        _fixed_filling_info(
            root=root,
            charge_integration_calls=charge_integration_calls,
            density_integration_calls=density_integration_calls,
            charge_work=charge_work,
            charge_refinements=charge_refinements,
            density_info=density_info,
            charge_tol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            num_threads=num_threads,
            band_energy=band_energy,
            nk=nk,
            charge_diagonalizations=charge_diagonalizations,
        ),
    )


__all__ = [
    "_ZERO_TEMP_EXT_AVAILABLE",
    "density_matrix_at_mu_zero_temp",
    "density_matrix_zero_temp",
]
