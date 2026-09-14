from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from meanfi.space.coordinates import DensityCoordinates


@dataclass(frozen=True)
class RequiredCoordinateSelection:
    coordinates: DensityCoordinates
    active_real_rows: np.ndarray
    value_real_rows: np.ndarray


def select_required_coordinates(
    active_coordinates: DensityCoordinates,
    basis: np.ndarray,
) -> RequiredCoordinateSelection:
    """Choose active entries whose real/imag values determine all parameters."""

    entries = active_coordinates.entries
    value_count = len(entries)
    parameter_count = basis.shape[1]
    if parameter_count:
        _, r, pivots = la.qr(basis.T, mode="economic", pivoting=True)
        tolerance = np.finfo(float).eps * max(basis.shape) * abs(r[0, 0])
        rank = np.sum(np.abs(np.diag(r)) > tolerance)
        if rank != parameter_count:
            raise ValueError(
                "Could not select enough active entries to determine params"
            )
        active_real_rows = np.asarray(pivots[:parameter_count], dtype=int)
    else:
        active_real_rows = np.empty(0, dtype=int)

    selected_positions = sorted({int(row % value_count) for row in active_real_rows})
    coordinates = DensityCoordinates.from_entries(
        size=active_coordinates.size,
        keys=list(active_coordinates.keys),
        entries=tuple(entries[position] for position in selected_positions),
    )
    selected_index = {
        entry: position for position, entry in enumerate(coordinates.entries)
    }
    value_real_rows = np.asarray(
        [
            selected_index[entries[row % value_count]]
            + (coordinates.value_count if row >= value_count else 0)
            for row in active_real_rows
        ],
        dtype=int,
    )
    return RequiredCoordinateSelection(coordinates, active_real_rows, value_real_rows)
