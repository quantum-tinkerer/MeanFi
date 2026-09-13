from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import as_sparse

from ..mumps_backend import SelectedInversePattern, build_selected_inverse_pattern


def _selection_requested_pairs(
    density_coordinates: DensityCoordinates,
) -> tuple[np.ndarray, np.ndarray]:
    stacked_rows = density_coordinates.all_rows
    stacked_cols = density_coordinates.all_cols
    if stacked_rows.size == 0:
        return stacked_rows, stacked_cols
    pairs = np.unique(np.stack([stacked_rows, stacked_cols], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


@dataclass(frozen=True)
class SparseChargePattern:
    pattern: SelectedInversePattern
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
    pattern: SelectedInversePattern
    value_positions: np.ndarray
    reverse_value_positions: np.ndarray
    value_is_diagonal: np.ndarray

    def density_values_from_inverse_entries(
        self,
        inverse_entries: dict[complex, np.ndarray],
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
    ) -> np.ndarray:
        values = np.zeros(self.value_positions.size, dtype=np.complex128)
        if self.value_positions.size == 0:
            return values

        values[self.value_is_diagonal] += complex(constant)
        for shift, residue in zip(shifts, residues, strict=True):
            pole_entries = inverse_entries[complex(shift)]
            values += residue * pole_entries[self.value_positions]
            values += np.conjugate(residue) * np.conjugate(
                pole_entries[self.reverse_value_positions]
            )
        return values


def build_sparse_charge_pattern(trace_weights_diag: np.ndarray) -> SparseChargePattern:
    weights = np.asarray(trace_weights_diag, dtype=float)
    diagonal = np.flatnonzero(np.abs(weights) > 0.0).astype(int, copy=False)
    pattern = build_selected_inverse_pattern(
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
    density_coordinates: DensityCoordinates,
) -> SparseDensityPattern:
    unique_rows, unique_cols = _selection_requested_pairs(density_coordinates)
    requested_rows = density_coordinates.all_rows
    requested_cols = density_coordinates.all_cols
    reverse_rows = unique_cols
    reverse_cols = unique_rows
    pattern = build_selected_inverse_pattern(
        size=size,
        rows=np.concatenate([unique_rows, reverse_rows]),
        cols=np.concatenate([unique_cols, reverse_cols]),
    )
    value_positions = np.asarray(
        [
            pattern.lookup[(int(row), int(col))]
            for row, col in zip(requested_rows, requested_cols, strict=True)
        ],
        dtype=int,
    )
    reverse_value_positions = np.asarray(
        [
            pattern.lookup[(int(row), int(col))]
            for row, col in zip(requested_cols, requested_rows, strict=True)
        ],
        dtype=int,
    )
    return SparseDensityPattern(
        pattern=pattern,
        value_positions=value_positions,
        reverse_value_positions=reverse_value_positions,
        value_is_diagonal=np.asarray(requested_rows == requested_cols, dtype=bool),
    )


def _pattern_subset_mappings(
    source: SelectedInversePattern,
    target: SelectedInversePattern,
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


def _sparse_shifted_matrix(matrix: Any, shift: complex):
    shifted = as_sparse(matrix).tocsc()
    shifted = shifted.copy()
    diagonal = np.asarray(shifted.diagonal(), dtype=complex)
    diagonal -= complex(shift)
    shifted.setdiag(diagonal)
    return shifted
