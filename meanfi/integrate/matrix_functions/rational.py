from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meanfi.core.matrix import as_sparse, is_sparse_like, sparse_linalg_module

from .base import RationalFOE, _BlockResult
from .common import _derivative_convergence


@lru_cache(maxsize=None)
def _ozaki_poles_and_residues(pole_count: int) -> tuple[np.ndarray, np.ndarray]:
    if pole_count <= 0:
        raise ValueError("pole_count must be positive")

    matrix_size = 2 * int(pole_count)
    diagonal_weights = np.arange(1, 2 * matrix_size, 2, dtype=float)
    off_diagonal = -0.5 / np.sqrt(diagonal_weights[:-1] * diagonal_weights[1:])
    continued_fraction = np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(continued_fraction)
    negative = eigenvalues < 0.0
    if int(np.count_nonzero(negative)) != int(pole_count):
        raise ValueError("Ozaki pole construction returned an unexpected pole count")

    selected_eigenvalues = eigenvalues[negative]
    selected_vectors = eigenvectors[0, negative]
    poles = -1.0 / selected_eigenvalues
    residues = -0.25 * np.square(selected_vectors) * np.square(poles)
    return np.asarray(poles, dtype=float), np.asarray(residues, dtype=float)


def _dense_shifted_matrix(matrix: np.ndarray, shift: float) -> np.ndarray:
    shifted = np.array(matrix, dtype=complex, copy=True)
    diagonal = shifted.diagonal().copy()
    diagonal -= 1j * float(shift)
    np.fill_diagonal(shifted, diagonal)
    return shifted


def _sparse_shifted_lu(matrix: Any, shift: float):
    shifted = as_sparse(matrix).tocsc().astype(complex)
    shifted = shifted.copy()
    diagonal = np.asarray(shifted.diagonal(), dtype=complex)
    diagonal -= 1j * float(shift)
    shifted.setdiag(diagonal)
    return sparse_linalg_module().splu(shifted)


def _evaluate_rational_poles(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    pole_count: int,
    derivative: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    poles, residues = _ozaki_poles_and_residues(pole_count)
    density_result = 0.5 * np.asarray(block, dtype=complex)
    derivative_block = np.zeros_like(block) if derivative else None

    if is_sparse_like(matrix):
        for pole, residue in zip(poles, residues, strict=True):
            shift = float(pole) * float(kT)
            lu = _sparse_shifted_lu(matrix, shift)
            y = np.asarray(lu.solve(block), dtype=complex)
            y_adj = np.asarray(lu.solve(block, trans="H"), dtype=complex)
            density_result = density_result + residue * kT * (y + y_adj)

            if derivative:
                rhs = q_diag[:, np.newaxis] * y
                rhs_adj = q_diag[:, np.newaxis] * y_adj
                z = np.asarray(lu.solve(rhs), dtype=complex)
                z_adj = np.asarray(lu.solve(rhs_adj, trans="H"), dtype=complex)
                derivative_block = derivative_block + residue * kT * (z + z_adj)

        return density_result, derivative_block

    dense_matrix = np.asarray(matrix, dtype=complex)
    for pole, residue in zip(poles, residues, strict=True):
        shift = float(pole) * float(kT)
        shifted = _dense_shifted_matrix(dense_matrix, shift)
        shifted_adjoint = shifted.conj().T
        y = np.linalg.solve(shifted, block)
        y_adj = np.linalg.solve(shifted_adjoint, block)
        density_result = density_result + residue * kT * (y + y_adj)

        if derivative:
            rhs = q_diag[:, np.newaxis] * y
            rhs_adj = q_diag[:, np.newaxis] * y_adj
            z = np.linalg.solve(shifted, rhs)
            z_adj = np.linalg.solve(shifted_adjoint, rhs_adj)
            derivative_block = derivative_block + residue * kT * (z + z_adj)

    return density_result, derivative_block


def _rational_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    options: RationalFOE,
    derivative_trace_monitor=None,
    derivative_context: str | None = None,
) -> _BlockResult:
    accepted_block = None
    accepted_derivative = None
    accepted_error = float("inf")
    accepted_order = None
    derivative_converged = True

    half_poles = int(options.initial_poles)
    half_block, half_derivative = _evaluate_rational_poles(
        matrix,
        block,
        kT=kT,
        q_diag=q_diag,
        pole_count=half_poles,
        derivative=derivative,
    )

    while 2 * half_poles <= int(options.max_poles):
        pole_count = 2 * half_poles
        full_block, full_derivative = _evaluate_rational_poles(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            pole_count=pole_count,
            derivative=derivative,
        )
        accepted_error = float(np.max(np.abs(full_block - half_block)))

        if derivative:
            derivative_converged, _derivative_error = _derivative_convergence(
                full_derivative,
                half_derivative,
                derivative=derivative,
                dn_dmu_rtol=options.dn_dmu_rtol,
                derivative_trace_monitor=derivative_trace_monitor,
                derivative_context=derivative_context,
                matrix_function_name="Rational FOE",
            )

        accepted_block = full_block
        accepted_derivative = full_derivative
        accepted_order = pole_count
        if accepted_error <= tolerance and derivative_converged:
            break

        half_poles = pole_count
        half_block = full_block
        half_derivative = full_derivative

    if accepted_error > tolerance or (derivative and not derivative_converged):
        raise ValueError("Rational FOE did not converge within max_poles")

    return _BlockResult(
        block=accepted_block,
        derivative_block=accepted_derivative,
        error=accepted_error,
        order=accepted_order,
    )
