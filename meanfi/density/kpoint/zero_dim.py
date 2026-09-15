"""Exact finite-system density for the zero-temperature simplex path."""

from __future__ import annotations

import numpy as np

from meanfi.results import DensityEntries, DensityResult
from meanfi.density.kpoint.matrix_functions.direct import (
    selected_density_values_from_eigensystem,
)
from meanfi.density.kpoint.occupations import fermi_dirac, occupation_entropy
from meanfi.errors import ErrorValues
from meanfi.results import FermiSimplexInfo
from meanfi.space.coordinates import DensityCoordinates


def evaluate_zero_dim(
    matrix: np.ndarray,
    coordinates: DensityCoordinates,
    *,
    mu: float | None = None,
    filling: float | None = None,
    mu_guess: float = 0.0,
    filling_tol: float = 1e-6,
    nk: int | None = None,
) -> DensityResult:
    """Diagonalize once, select occupations, and return the requested entries."""

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if filling is not None:
        mu, charge = zero_dim_zero_temp_mu(
            eigenvalues, filling=filling, mu_guess=mu_guess
        )
        if abs(charge - filling) > filling_tol:
            raise RuntimeError(
                "The zero-dimensional spectrum cannot represent the requested filling"
            )
    occupation = fermi_dirac(eigenvalues, 0.0, mu)
    charge = float(np.sum(occupation))
    # Full layouts use a matrix product; selected layouts avoid the full density.
    if coordinates.is_full:
        density = (eigenvectors * occupation) @ eigenvectors.conj().T
        values = coordinates.values_from_assembled_matrix(density)
    else:
        values = selected_density_values_from_eigensystem(
            eigenvectors, occupation, coordinates
        )
    estimated = nk is None
    error = 0.0 if estimated else None
    info = FermiSimplexInfo(
        n_kernel_evals=1,
        unique_evals=1,
        n_evaluator_evals=1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        refinements=0,
        error_estimate_available=estimated,
        charge_evaluations=0 if filling is None else 1,
        charge_integration_calls=0,
        density_integration_calls=1,
        requested_nk=nk,
        n_kpoints=1,
        n_diagonalizations=1,
    )
    return DensityResult(
        entries=DensityEntries(
            coordinates,
            values,
            np.zeros(coordinates.value_count) if estimated else None,
        ),
        mu=float(mu),
        filling=charge,
        errors=ErrorValues(
            density_matrix_integration=error,
            charge_integration=error,
            band_energy_integration=error,
            entropy_integration=error,
            filling_residual=None if filling is None else abs(charge - filling),
        ),
        statistics=info,
        band_energy=float(eigenvalues @ occupation) / eigenvalues.size,
        entropy=float(occupation_entropy(occupation).mean()),
    )


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
