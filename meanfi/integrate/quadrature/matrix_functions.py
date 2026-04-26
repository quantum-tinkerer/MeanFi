from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import warnings

import numpy as np

from meanfi.core.matrix import (
    as_sparse,
    conjugate_transpose,
    is_sparse_like,
    matrix_shape,
    sparse_linalg_module,
    sparse_module,
    to_dense,
)

from .normal_backend import fermi_dirac

__all__ = [
    "BdGMatrixFunction",
    "ChebyshevFOE",
    "DirectDiagonalization",
    "RationalFOE",
    "basis_block",
    "density_block",
    "gershgorin_bounds",
    "matrix_function_label",
    "shift_by_mu",
]


@dataclass(frozen=True)
class BdGMatrixFunction:
    """Base class for BdG matrix-function density evaluators."""


@dataclass(frozen=True)
class DirectDiagonalization(BdGMatrixFunction):
    """Evaluate the finite-temperature BdG density matrix by diagonalization."""


@dataclass(frozen=True)
class ChebyshevFOE(BdGMatrixFunction):
    """Evaluate BdG density block-columns with a Chebyshev Fermi-operator expansion."""

    initial_order: int = 16
    max_order: int = 1024
    coefficient_oversampling: int = 4
    spectral_padding: float = 1e-8
    dn_dmu_rtol: float = 1e-1

    def __post_init__(self) -> None:
        if self.initial_order <= 0:
            raise ValueError("initial_order must be positive")
        if self.max_order < 2 * self.initial_order:
            raise ValueError("max_order must be at least twice initial_order")
        if self.coefficient_oversampling <= 0:
            raise ValueError("coefficient_oversampling must be positive")
        if self.spectral_padding < 0:
            raise ValueError("spectral_padding must be non-negative")
        if self.dn_dmu_rtol < 0:
            raise ValueError("dn_dmu_rtol must be non-negative")


@dataclass(frozen=True)
class RationalFOE(BdGMatrixFunction):
    """Evaluate BdG density block-columns with an Ozaki rational Fermi-operator expansion."""

    initial_poles: int = 4
    max_poles: int = 256
    dn_dmu_rtol: float = 1e-1

    def __post_init__(self) -> None:
        if self.initial_poles <= 0:
            raise ValueError("initial_poles must be positive")
        if self.max_poles < 2 * self.initial_poles:
            raise ValueError("max_poles must be at least twice initial_poles")
        if self.dn_dmu_rtol < 0:
            raise ValueError("dn_dmu_rtol must be non-negative")


@dataclass(frozen=True)
class _BlockResult:
    block: np.ndarray
    derivative_block: np.ndarray | None
    error: float
    order: int | None


_DN_DMU_ABS_FLOOR = 1e-6


def basis_block(size: int, columns: Sequence[int]) -> np.ndarray:
    block = np.zeros((size, len(columns)), dtype=complex)
    block[np.asarray(columns, dtype=int), np.arange(len(columns))] = 1.0
    return block


def matrix_action(matrix: Any, block: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ block, dtype=complex)


def shift_by_mu(matrix: Any, mu: float, q_diag: np.ndarray):
    if is_sparse_like(matrix):
        sparse = sparse_module()
        return matrix - float(mu) * sparse.diags(q_diag, format="csr")
    return np.asarray(matrix, dtype=complex) - float(mu) * np.diag(q_diag)


def gershgorin_bounds(matrix: Any) -> tuple[float, float]:
    if is_sparse_like(matrix):
        diagonal = np.asarray(matrix.diagonal(), dtype=complex)
        row_sums = np.asarray(abs(matrix).sum(axis=1)).ravel()
    else:
        array = np.asarray(matrix, dtype=complex)
        diagonal = np.diag(array)
        row_sums = np.sum(np.abs(array), axis=1)

    radius = np.maximum(row_sums - np.abs(diagonal), 0.0)
    center = diagonal.real
    return float(np.min(center - radius)), float(np.max(center + radius))


def matrix_function_label(matrix_function: BdGMatrixFunction) -> str:
    if isinstance(matrix_function, ChebyshevFOE):
        return "Chebyshev FOE"
    if isinstance(matrix_function, RationalFOE):
        return "Rational FOE"
    if isinstance(matrix_function, DirectDiagonalization):
        return "direct diagonalization"
    return matrix_function.__class__.__name__


def _derivative_convergence(
    derivative_full: np.ndarray,
    derivative_half: np.ndarray,
    *,
    derivative: bool,
    dn_dmu_rtol: float,
    derivative_trace_monitor,
    derivative_context: str | None,
    matrix_function_name: str,
) -> tuple[bool, float]:
    if not derivative:
        return True, 0.0

    if derivative_trace_monitor is None:
        derivative_error = float(np.max(np.abs(derivative_full - derivative_half)))
        derivative_scale = max(
            float(np.max(np.abs(derivative_full))),
            float(np.max(np.abs(derivative_half))),
            1e-15,
        )
        return derivative_error <= float(dn_dmu_rtol) * derivative_scale, derivative_error

    derivative_full_trace = float(derivative_trace_monitor(derivative_full))
    derivative_half_trace = float(derivative_trace_monitor(derivative_half))
    derivative_scale = max(
        abs(derivative_full_trace),
        abs(derivative_half_trace),
        _DN_DMU_ABS_FLOOR,
    )
    derivative_error = abs(derivative_full_trace - derivative_half_trace)
    if (
        abs(derivative_full_trace) <= _DN_DMU_ABS_FLOOR
        and abs(derivative_half_trace) <= _DN_DMU_ABS_FLOOR
    ):
        warnings.warn(
            f"BdG {matrix_function_name} dn/dmu reached absolute floor; treating as singular local point"
            + ("" if derivative_context is None else f" ({derivative_context})"),
            RuntimeWarning,
        )
        return True, derivative_error
    return derivative_error <= float(dn_dmu_rtol) * derivative_scale, derivative_error


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


def _chebyshev_coefficients(
    order: int,
    *,
    center: float,
    scale: float,
    kT: float,
    oversampling: int,
) -> np.ndarray:
    n_nodes = max(64, int(oversampling) * (order + 1))
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    values = fermi_dirac(scale * np.cos(theta) + center, kT, 0.0)
    coeffs = np.empty(order + 1, dtype=float)
    coeffs[0] = np.mean(values)
    for mode in range(1, order + 1):
        coeffs[mode] = 2.0 * np.mean(values * np.cos(mode * theta))
    return coeffs


def _exact_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
) -> _BlockResult:
    array = to_dense(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    occupation = fermi_dirac(eigenvalues, kT, 0.0)
    projected_block = eigenvectors.conj().T @ block
    density_result = eigenvectors @ (occupation[:, np.newaxis] * projected_block)

    derivative_block = None
    if derivative:
        fprime = -occupation * (1.0 - occupation) / kT
        delta = eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]
        numerator = occupation[:, np.newaxis] - occupation[np.newaxis, :]
        loewner = np.empty_like(delta, dtype=float)
        separated = np.abs(delta) > 1e-12
        loewner[separated] = numerator[separated] / delta[separated]
        loewner[~separated] = np.broadcast_to(
            fprime[:, np.newaxis], delta.shape
        )[~separated]

        d_h = -(q_diag[:, np.newaxis] * eigenvectors)
        projected_dh = eigenvectors.conj().T @ d_h
        derivative_block = eigenvectors @ ((loewner * projected_dh) @ projected_block)

    return _BlockResult(
        block=density_result,
        derivative_block=derivative_block,
        error=0.0,
        order=None,
    )


def _apply_rescaled(matrix: Any, block: np.ndarray, *, center: float, scale: float) -> np.ndarray:
    return (matrix_action(matrix, block) - center * block) / scale


def _chebyshev_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    options: ChebyshevFOE,
    derivative_trace_monitor=None,
    derivative_context: str | None = None,
) -> _BlockResult:
    lower, upper = gershgorin_bounds(matrix)
    width = max(upper - lower, 1e-12)
    scale = 0.5 * width * (1.0 + options.spectral_padding)
    center = 0.5 * (upper + lower)

    accepted_block = None
    accepted_derivative = None
    accepted_error = float("inf")
    accepted_order = None
    derivative_converged = True

    order_half = int(options.initial_order)
    while 2 * order_half <= int(options.max_order):
        order = 2 * order_half
        coeffs = _chebyshev_coefficients(
            order,
            center=center,
            scale=scale,
            kT=kT,
            oversampling=options.coefficient_oversampling,
        )

        v_prev = np.array(block, copy=True)
        v_curr = _apply_rescaled(matrix, block, center=center, scale=scale)
        y_half = coeffs[0] * v_prev
        y_full = coeffs[0] * v_prev

        if derivative:
            s_prev = np.zeros_like(block)
            s_curr = -(q_diag[:, np.newaxis] * block) / scale
            dy_half = coeffs[0] * s_prev
            dy_full = coeffs[0] * s_prev
        else:
            s_prev = s_curr = None
            dy_half = dy_full = None

        if order_half >= 1:
            y_half = y_half + coeffs[1] * v_curr
        y_full = y_full + coeffs[1] * v_curr
        if derivative:
            if order_half >= 1:
                dy_half = dy_half + coeffs[1] * s_curr
            dy_full = dy_full + coeffs[1] * s_curr

        for mode in range(1, order):
            v_next = 2.0 * _apply_rescaled(
                matrix,
                v_curr,
                center=center,
                scale=scale,
            ) - v_prev
            if derivative:
                d_x_v = -(q_diag[:, np.newaxis] * v_curr) / scale
                s_next = (
                    2.0
                    * _apply_rescaled(
                        matrix,
                        s_curr,
                        center=center,
                        scale=scale,
                    )
                    + 2.0 * d_x_v
                    - s_prev
                )

            term_index = mode + 1
            if term_index <= order_half:
                y_half = y_half + coeffs[term_index] * v_next
                if derivative:
                    dy_half = dy_half + coeffs[term_index] * s_next
            y_full = y_full + coeffs[term_index] * v_next
            if derivative:
                dy_full = dy_full + coeffs[term_index] * s_next

            v_prev, v_curr = v_curr, v_next
            if derivative:
                s_prev, s_curr = s_curr, s_next

        accepted_error = float(np.max(np.abs(y_full - y_half)))
        if derivative:
            derivative_converged, _derivative_error = _derivative_convergence(
                dy_full,
                dy_half,
                derivative=derivative,
                dn_dmu_rtol=options.dn_dmu_rtol,
                derivative_trace_monitor=derivative_trace_monitor,
                derivative_context=derivative_context,
                matrix_function_name="Chebyshev",
            )
        accepted_block = y_full
        accepted_derivative = dy_full
        accepted_order = order
        if accepted_error <= tolerance and derivative_converged:
            break
        order_half = order

    if accepted_error > tolerance or (derivative and not derivative_converged):
        raise ValueError("Chebyshev FOE did not converge within max_order")

    return _BlockResult(
        block=accepted_block,
        derivative_block=accepted_derivative,
        error=accepted_error,
        order=accepted_order,
    )


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


def density_block(
    matrix_function: BdGMatrixFunction,
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    derivative_trace_monitor=None,
    derivative_context: str | None = None,
) -> _BlockResult:
    if isinstance(matrix_function, DirectDiagonalization):
        return _exact_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
        )
    if isinstance(matrix_function, ChebyshevFOE):
        return _chebyshev_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
            tolerance=tolerance,
            options=matrix_function,
            derivative_trace_monitor=derivative_trace_monitor,
            derivative_context=derivative_context,
        )
    if isinstance(matrix_function, RationalFOE):
        return _rational_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
            tolerance=tolerance,
            options=matrix_function,
            derivative_trace_monitor=derivative_trace_monitor,
            derivative_context=derivative_context,
        )
    raise TypeError("matrix_function must be a BdGMatrixFunction instance")
