from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import as_sparse

from ..mumps_backend import SelectedInversePattern, build_selected_inverse_pattern


@dataclass(frozen=True)
class SparseRationalLayout:
    """Fixed inverse entries and their mapping to requested density coordinates."""

    charge: SelectedInversePattern
    density: SelectedInversePattern
    extra: SelectedInversePattern
    charge_weights: np.ndarray
    value_positions: np.ndarray
    reverse_value_positions: np.ndarray
    value_is_diagonal: np.ndarray
    charge_positions: np.ndarray
    density_charge_positions: np.ndarray
    density_extra_positions: np.ndarray

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    @classmethod
    def build(
        cls,
        *,
        density_coordinates: DensityCoordinates,
        trace_weights_diag: np.ndarray,
        include_all_diagonal: bool = False,
    ) -> SparseRationalLayout:
        weights = np.asarray(trace_weights_diag, dtype=float)
        size = density_coordinates.size
        if weights.shape != (size,):
            raise ValueError("Trace weights must match the density coordinate size")
        diagonal = np.arange(size) if include_all_diagonal else np.flatnonzero(weights)
        charge = build_selected_inverse_pattern(size=size, rows=diagonal, cols=diagonal)
        rows, cols = density_coordinates.all_rows, density_coordinates.all_cols
        density = build_selected_inverse_pattern(
            size=size,
            rows=np.concatenate([rows, cols]),
            cols=np.concatenate([cols, rows]),
        )
        # Both patterns use CSC order, so filtering preserves the extra pattern's
        # order and every density entry belongs to exactly one source.
        charge_positions = np.array(
            [charge.lookup.get(pair, -1) for pair in density.lookup], dtype=int
        )
        from_charge = charge_positions >= 0
        extra = build_selected_inverse_pattern(
            size=size, rows=density.rows[~from_charge], cols=density.cols[~from_charge]
        )
        return cls(
            charge=charge,
            density=density,
            extra=extra,
            charge_weights=weights[diagonal],
            value_positions=np.array(
                [
                    density.lookup[(int(row), int(col))]
                    for row, col in zip(rows, cols, strict=True)
                ],
                dtype=int,
            ),
            reverse_value_positions=np.array(
                [
                    density.lookup[(int(col), int(row))]
                    for row, col in zip(rows, cols, strict=True)
                ],
                dtype=int,
            ),
            value_is_diagonal=rows == cols,
            charge_positions=charge_positions[from_charge],
            density_charge_positions=np.flatnonzero(from_charge),
            density_extra_positions=np.flatnonzero(~from_charge),
        )

    def charge_from_inverse_entries(
        self,
        inverse_entries: dict[complex, np.ndarray],
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
    ) -> float:
        diagonal = np.full(self.charge.nnz, complex(constant), dtype=np.complex128)
        for shift, residue in zip(shifts, residues, strict=True):
            entries = inverse_entries[complex(shift)]
            diagonal += residue * entries + np.conjugate(residue * entries)
        return float(np.real(self.charge_weights @ diagonal))

    def density_values_from_inverse_entries(
        self,
        inverse_entries: dict[complex, np.ndarray],
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
    ) -> np.ndarray:
        values = np.zeros(self.value_positions.size, dtype=np.complex128)
        values[self.value_is_diagonal] = complex(constant)
        for shift, residue in zip(shifts, residues, strict=True):
            entries = inverse_entries[complex(shift)]
            values += residue * entries[self.value_positions]
            values += np.conjugate(residue * entries[self.reverse_value_positions])
        return values


@dataclass(frozen=True)
class SparseRationalTerms:
    constant: complex
    shifts: np.ndarray
    residues: np.ndarray
    pole_count: int
    entropy_constant: complex | None = None
    entropy_residues: np.ndarray | None = None
    entropy_error: float | None = None


def _sparse_shifted_matrix(matrix: Any, shift: complex):
    shifted = as_sparse(matrix).tocsc().copy()
    diagonal = np.asarray(shifted.diagonal(), dtype=complex)
    diagonal -= complex(shift)
    shifted.setdiag(diagonal)
    return shifted
