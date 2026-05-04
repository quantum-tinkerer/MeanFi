from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np

from meanfi.tb.ops import hermitian_spectral_bound, is_sparse_like, sparse_module
_DN_DMU_ABS_FLOOR = 1e-6


def basis_block(size: int, columns: Sequence[int], *, dtype: np.dtype = np.dtype(complex)) -> np.ndarray:
    block = np.zeros((size, len(columns)), dtype=dtype)
    block[np.asarray(columns, dtype=int), np.arange(len(columns))] = 1.0
    return block


def matrix_action(matrix: Any, block: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ block)


def workspace_matrix(matrix: Any, dtype: np.dtype):
    if is_sparse_like(matrix):
        return matrix.astype(dtype, copy=False)
    return np.asarray(matrix, dtype=dtype)


def shift_by_mu(matrix: Any, mu: float, q_diag: np.ndarray, *, dtype: np.dtype | None = None):
    resolved_dtype = np.dtype(complex) if dtype is None else np.dtype(dtype)
    if is_sparse_like(matrix):
        sparse = sparse_module()
        shifted = matrix.astype(resolved_dtype, copy=False)
        return shifted - float(mu) * sparse.diags(q_diag.astype(resolved_dtype), format="csr")
    return np.asarray(matrix, dtype=resolved_dtype) - float(mu) * np.diag(
        q_diag.astype(resolved_dtype)
    )


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


def spectral_interval(
    matrix: Any,
    *,
    spectral_padding: float = 0.0,
) -> tuple[float, float]:
    lower, upper = gershgorin_bounds(matrix)
    spectral_bound = hermitian_spectral_bound(matrix)
    if spectral_bound <= 0.0:
        return lower, upper
    center = 0.5 * (lower + upper)
    half_width = max(spectral_bound, abs(lower - center), abs(upper - center), 1e-12)
    padded = half_width * (1.0 + max(0.0, float(spectral_padding)))
    return center - padded, center + padded


def scalar_derivative_converged(full: float, half: float, *, rtol: float) -> bool:
    scale = max(abs(full), abs(half), _DN_DMU_ABS_FLOOR)
    if abs(full) <= _DN_DMU_ABS_FLOOR and abs(half) <= _DN_DMU_ABS_FLOOR:
        return True
    return abs(full - half) <= float(rtol) * scale


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
