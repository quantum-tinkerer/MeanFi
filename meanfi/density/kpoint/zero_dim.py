from __future__ import annotations

import numpy as np

from meanfi.density.filling import (
    charge_integral_tolerance,
    mu_bracket,
    solve_mu,
)
from meanfi.density.kpoint.matrix_functions import (
    RationalFOE,
    resolve_matrix_function,
)
from meanfi.density.kpoint.occupations import fermi_dirac
from meanfi.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.tb.ops import _tb_type
from meanfi.space.coordinates import DensityCoordinates


def density_from_matrix(
    matrix: np.ndarray,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
) -> tuple[_tb_type, _tb_type]:
    """Embed a local density matrix into the requested tight-binding keys."""

    if density_coordinates is not None:
        values = density_coordinates.values_from_assembled_matrix(matrix)
        zeros = np.zeros(density_coordinates.value_count, dtype=float)
        return density_coordinates.values_and_errors_to_tb(values, zeros)

    onsite_key = tuple(0 for _ in next(iter(keys), tuple()))
    rho = {key: matrix if key == onsite_key else np.zeros_like(matrix) for key in keys}
    rho_error = {key: np.zeros_like(matrix) for key in keys}
    return rho, rho_error


def density_matrix_at_mu_zero_dim(
    matrix: np.ndarray,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
    matrix_function: object | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
    density_tolerance: float = 0.0,
) -> tuple[_tb_type, _tb_type, DensityIntegrationInfo]:
    """Evaluate the density matrix for a finite system at fixed chemical potential."""

    resolved_matrix_function = resolve_matrix_function(matrix_function)
    if isinstance(resolved_matrix_function, RationalFOE):
        raise ValueError(
            "Dense RationalFOE is unsupported; use DirectDiagonalization for dense matrices "
            "or sparse matrices for the MUMPS selected-inverse RationalFOE path"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    occupation = fermi_dirac(eigenvalues, kT, mu)
    density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
    rho, error = density_from_matrix(density, keys, density_coordinates)
    info = DensityIntegrationInfo(
        n_kernel_evals=1,
        unique_evals=1,
        n_evaluator_evals=1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
        error_estimate_available=True,
    )
    return rho, error, info


def density_matrix_zero_dim(
    matrix: np.ndarray,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    mu_guess: float,
    charge_tol: float,
    mu_xtol: float,
    max_charge_evaluations: int | None,
    density_atol: float,
    density_rtol: float,
    density_coordinates: DensityCoordinates | None = None,
    matrix_function: object | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> tuple[_tb_type, _tb_type, float, FixedFillingInfo]:
    """Evaluate the fixed-filling density matrix for a finite system."""

    resolved_matrix_function = resolve_matrix_function(matrix_function)
    eigenvalues = eigenvectors = None
    if not isinstance(resolved_matrix_function, RationalFOE):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if kT == 0:
        assert eigenvalues is not None
        assert eigenvectors is not None
        mu, charge = zero_dim_zero_temp_mu(
            eigenvalues,
            filling=filling,
            mu_guess=mu_guess,
        )
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
        rho, error = density_from_matrix(density, keys, density_coordinates)
        info = FixedFillingInfo(
            mu=mu,
            charge=charge,
            charge_error=abs(charge - filling),
            dcharge_dmu=0.0,
            charge_evaluations=1,
            charge_integration_calls=1,
            density_integration_calls=1,
            charge_n_kernel_evals=1,
            density_n_kernel_evals=0,
            n_kernel_evals=1,
            unique_evals=1,
            charge_n_evaluator_evals=1,
            density_n_evaluator_evals=1,
            n_evaluator_evals=2,
            n_cached_nodes=1,
            n_leaves=1,
            n_leaf_nodes=1,
            subdivisions=0,
            charge_integral_atol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            error_estimate_available=True,
        )
        return rho, error, mu, info

    if isinstance(resolved_matrix_function, RationalFOE):
        raise ValueError(
            "Dense RationalFOE is unsupported; use DirectDiagonalization for dense matrices "
            "or sparse matrices for the MUMPS selected-inverse RationalFOE path"
        )

    assert eigenvalues is not None
    assert eigenvectors is not None
    charge_calls = 0

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        nonlocal charge_calls
        charge_calls += 1
        occupation = fermi_dirac(eigenvalues, kT, mu)
        charge = float(np.sum(occupation))
        derivative = float(np.sum(occupation * (1.0 - occupation) / kT))
        return charge, 0.0, derivative

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=lambda: mu_bracket({tuple(): matrix}, kT),
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=charge_tol,
        mu_tol=mu_xtol,
        max_charge_evaluations=max_charge_evaluations,
    )

    occupation = fermi_dirac(eigenvalues, kT, root.mu)
    density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
    rho, error = density_from_matrix(density, keys, density_coordinates)
    charge_integral_atol, _ = charge_integral_tolerance(charge_tol)
    info = FixedFillingInfo(
        mu=root.mu,
        charge=root.charge,
        charge_error=root.charge_error,
        dcharge_dmu=float("nan") if root.derivative is None else root.derivative,
        charge_evaluations=root.charge_evaluations,
        charge_integration_calls=charge_calls,
        density_integration_calls=1,
        charge_n_kernel_evals=1,
        density_n_kernel_evals=0,
        n_kernel_evals=1,
        unique_evals=1,
        charge_n_evaluator_evals=charge_calls,
        density_n_evaluator_evals=1,
        n_evaluator_evals=charge_calls + 1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=density_rtol,
        error_estimate_available=True,
    )
    return rho, error, root.mu, info


def zero_dim_zero_temp_mu(
    eigenvalues: np.ndarray,
    *,
    filling: float,
    mu_guess: float,
) -> tuple[float, float]:
    """Pick the zero-temperature chemical potential closest to the target filling."""

    unique = np.unique(np.asarray(eigenvalues, dtype=float))
    candidates: list[tuple[float, bool]] = []
    if unique.size == 0:
        return 0.0, 0.0

    candidates.append((unique[0] - 1.0, False))
    candidates.extend((float(energy), False) for energy in unique)
    candidates.extend(
        (0.5 * float(left + right), True)
        for left, right in zip(unique[:-1], unique[1:], strict=False)
    )
    candidates.append((unique[-1] + 1.0, False))

    best_mu = float(mu_guess)
    best_charge = float(np.sum(fermi_dirac(eigenvalues, 0.0, best_mu)))
    best_key = (abs(best_charge - filling), 1, abs(best_mu - mu_guess))
    for candidate_mu, is_midgap in candidates:
        charge = float(np.sum(fermi_dirac(eigenvalues, 0.0, candidate_mu)))
        candidate_key = (
            abs(charge - filling),
            0 if is_midgap else 1,
            abs(candidate_mu - mu_guess),
        )
        if candidate_key < best_key:
            best_key = candidate_key
            best_mu = float(candidate_mu)
            best_charge = charge
    return best_mu, best_charge
