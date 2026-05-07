from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse.linalg as sparse_linalg

from meanfi.state.support import DensityEntrySupport
from meanfi.tb.ops import as_sparse, is_sparse_like

from ..base import RationalFOE, _BlockResult
from ..common import (
    _derivative_convergence,
    spectral_interval,
    workspace_matrix,
)
from ..mumps_backend import MumpsSelectedEntryPattern, build_selected_entry_pattern


def _support_requested_pairs(
    density_support: DensityEntrySupport,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for block_rows, block_cols in zip(
        density_support.row_indices,
        density_support.col_indices,
        strict=True,
    ):
        if block_rows.size == 0:
            continue
        rows.append(np.asarray(block_rows, dtype=int))
        cols.append(np.asarray(block_cols, dtype=int))
    if not rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    stacked_rows = np.concatenate(rows)
    stacked_cols = np.concatenate(cols)
    pairs = np.unique(np.stack([stacked_rows, stacked_cols], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


@dataclass(frozen=True)
class SparseChargePattern:
    pattern: MumpsSelectedEntryPattern
    diagonal_positions: np.ndarray
    charge_weights: np.ndarray

    def charge_from_inverse_entries(
        self,
        inverse_entries: dict[complex, np.ndarray],
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
    ) -> float:
        diagonal = np.full(
            self.diagonal_positions.size, complex(constant), dtype=np.complex128
        )
        for shift, residue in zip(shifts, residues, strict=True):
            pole_entries = inverse_entries[complex(shift)]
            diagonal += residue * pole_entries[self.diagonal_positions]
            diagonal += np.conjugate(residue) * np.conjugate(
                pole_entries[self.diagonal_positions]
            )
        return float(np.real(np.sum(self.charge_weights * diagonal)))


@dataclass(frozen=True)
class SparseDensityPattern:
    pattern: MumpsSelectedEntryPattern
    requested_rows: np.ndarray
    requested_cols: np.ndarray
    requested_col_positions: np.ndarray
    requested_positions: np.ndarray
    requested_reverse_positions: np.ndarray
    requested_is_diagonal: np.ndarray
    selected_columns: np.ndarray

    def density_columns_from_inverse_entries(
        self,
        inverse_entries: dict[complex, np.ndarray],
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
    ) -> np.ndarray:
        columns = np.zeros(
            (self.pattern.size, self.selected_columns.size),
            dtype=np.complex128,
        )
        if self.requested_positions.size == 0:
            return columns

        values = np.zeros(self.requested_positions.size, dtype=np.complex128)
        values[self.requested_is_diagonal] += complex(constant)
        for shift, residue in zip(shifts, residues, strict=True):
            pole_entries = inverse_entries[complex(shift)]
            values += residue * pole_entries[self.requested_positions]
            values += np.conjugate(residue) * np.conjugate(
                pole_entries[self.requested_reverse_positions]
            )
        columns[self.requested_rows, self.requested_col_positions] = values
        return columns


def build_sparse_charge_pattern(trace_weights_diag: np.ndarray) -> SparseChargePattern:
    weights = np.asarray(trace_weights_diag, dtype=float)
    diagonal = np.flatnonzero(np.abs(weights) > 0.0).astype(int, copy=False)
    pattern = build_selected_entry_pattern(
        size=weights.size, rows=diagonal, cols=diagonal
    )
    diagonal_positions = np.asarray(
        [pattern.lookup[(int(index), int(index))] for index in diagonal],
        dtype=int,
    )
    return SparseChargePattern(
        pattern=pattern,
        diagonal_positions=diagonal_positions,
        charge_weights=weights[diagonal],
    )


def build_sparse_density_pattern(
    *,
    size: int,
    density_support: DensityEntrySupport,
) -> SparseDensityPattern:
    requested_rows, requested_cols = _support_requested_pairs(density_support)
    reverse_rows = requested_cols
    reverse_cols = requested_rows
    pattern = build_selected_entry_pattern(
        size=size,
        rows=np.concatenate([requested_rows, reverse_rows]),
        cols=np.concatenate([requested_cols, reverse_cols]),
    )
    selected_lookup = np.full(size, -1, dtype=int)
    selected_lookup[np.asarray(density_support.selected_columns, dtype=int)] = (
        np.arange(
            density_support.selected_columns.size,
            dtype=int,
        )
    )
    requested_positions = np.asarray(
        [
            pattern.lookup[(int(row), int(col))]
            for row, col in zip(requested_rows, requested_cols, strict=True)
        ],
        dtype=int,
    )
    requested_reverse_positions = np.asarray(
        [
            pattern.lookup[(int(row), int(col))]
            for row, col in zip(reverse_rows, reverse_cols, strict=True)
        ],
        dtype=int,
    )
    requested_col_positions = (
        selected_lookup[requested_cols]
        if requested_cols.size
        else np.empty(0, dtype=int)
    )
    return SparseDensityPattern(
        pattern=pattern,
        requested_rows=np.asarray(requested_rows, dtype=int),
        requested_cols=np.asarray(requested_cols, dtype=int),
        requested_col_positions=np.asarray(requested_col_positions, dtype=int),
        requested_positions=requested_positions,
        requested_reverse_positions=requested_reverse_positions,
        requested_is_diagonal=np.asarray(requested_rows == requested_cols, dtype=bool),
        selected_columns=np.asarray(density_support.selected_columns, dtype=int),
    )


def _pattern_subset_mappings(
    source: MumpsSelectedEntryPattern,
    target: MumpsSelectedEntryPattern,
) -> tuple[np.ndarray, np.ndarray]:
    source_positions: list[int] = []
    target_positions: list[int] = []
    for pair, source_position in source.lookup.items():
        target_position = target.lookup.get(pair)
        if target_position is None:
            continue
        source_positions.append(int(source_position))
        target_positions.append(int(target_position))
    return (
        np.asarray(source_positions, dtype=int),
        np.asarray(target_positions, dtype=int),
    )


@dataclass(frozen=True)
class SparseRationalTerms:
    constant: complex
    shifts: np.ndarray
    residues: np.ndarray
    pole_count: int
    support_count: int | None = None
    tail_lower_bound: float | None = None
    tail_upper_bound: float | None = None


def _dense_shifted_matrix(matrix: np.ndarray, shift: complex) -> np.ndarray:
    shifted = np.array(matrix, copy=True)
    diagonal = shifted.diagonal().copy()
    diagonal -= complex(shift)
    np.fill_diagonal(shifted, diagonal)
    return shifted


def _sparse_shifted_matrix(matrix: Any, shift: complex):
    shifted = as_sparse(matrix).tocsc()
    shifted = shifted.copy()
    diagonal = np.asarray(shifted.diagonal(), dtype=complex)
    diagonal -= complex(shift)
    shifted.setdiag(diagonal)
    return shifted


def _sparse_shifted_lu(matrix: Any, shift: complex):
    return sparse_linalg.splu(_sparse_shifted_matrix(matrix, shift))


def _evaluate_rational_terms(
    matrix: Any,
    block: np.ndarray,
    *,
    constant: complex,
    shifts: np.ndarray,
    residues: np.ndarray,
    q_diag: np.ndarray,
    derivative: bool,
    workspace_dtype: np.dtype,
    lu_cache: dict[complex, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[complex, Any] | None]:
    matrix = workspace_matrix(matrix, workspace_dtype)
    block = np.asarray(block, dtype=workspace_dtype)
    density_result = np.asarray(constant * block, dtype=complex)
    derivative_block = np.zeros_like(block, dtype=complex) if derivative else None

    if is_sparse_like(matrix):
        active_cache = {} if lu_cache is None else dict(lu_cache)
        for shift, residue in zip(shifts, residues, strict=True):
            key = complex(shift)
            lu = active_cache.get(key)
            if lu is None:
                lu = _sparse_shifted_lu(matrix, key)
                active_cache[key] = lu
            rhs = np.asarray(block, dtype=workspace_dtype)
            y = np.asarray(lu.solve(rhs), dtype=complex)
            y_adj = np.asarray(
                lu.solve(np.asarray(block, dtype=workspace_dtype), trans="H"),
                dtype=complex,
            )
            density_result = (
                density_result + residue * y + np.conjugate(residue) * y_adj
            )

            if derivative:
                rhs = np.asarray(q_diag[:, np.newaxis] * y, dtype=workspace_dtype)
                rhs_adj = np.asarray(
                    q_diag[:, np.newaxis] * y_adj, dtype=workspace_dtype
                )
                z = np.asarray(lu.solve(rhs), dtype=complex)
                z_adj = np.asarray(lu.solve(rhs_adj, trans="H"), dtype=complex)
                derivative_block = (
                    derivative_block + residue * z + np.conjugate(residue) * z_adj
                )
        return density_result, derivative_block, active_cache

    dense_matrix = np.asarray(matrix, dtype=workspace_dtype)
    for shift, residue in zip(shifts, residues, strict=True):
        shifted = _dense_shifted_matrix(dense_matrix, shift)
        shifted_adjoint = shifted.conj().T
        y = np.linalg.solve(shifted, block)
        y_adj = np.linalg.solve(shifted_adjoint, block)
        density_result = density_result + residue * y + np.conjugate(residue) * y_adj

        if derivative:
            rhs = q_diag[:, np.newaxis] * y
            rhs_adj = q_diag[:, np.newaxis] * y_adj
            z = np.linalg.solve(shifted, rhs)
            z_adj = np.linalg.solve(shifted_adjoint, rhs_adj)
            derivative_block = (
                derivative_block + residue * z + np.conjugate(residue) * z_adj
            )

    return density_result, derivative_block, lu_cache


def _evaluate_rational_poles(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    pole_count: int,
    derivative: bool,
    options: RationalFOE,
    workspace_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray | None]:
    from .scheme import _scheme_terms

    lower, upper = spectral_interval(matrix)
    constant, shifts, residues = _scheme_terms(
        options,
        pole_count,
        lower=lower,
        upper=upper,
        kT=kT,
    )
    density_result, derivative_block, _ = _evaluate_rational_terms(
        matrix,
        block,
        constant=constant,
        shifts=shifts,
        residues=residues,
        q_diag=q_diag,
        derivative=derivative,
        workspace_dtype=workspace_dtype,
    )
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
    workspace_dtype: np.dtype = np.dtype(complex),
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
        options=options,
        workspace_dtype=workspace_dtype,
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
            options=options,
            workspace_dtype=workspace_dtype,
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
