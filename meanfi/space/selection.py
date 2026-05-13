from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from meanfi.space.coordinates import DensityCoordinates, DensityEntry


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

    selected = _select_required_entries(active_coordinates.entries, basis)
    required_coordinates = _coordinates_from_entries(
        size=active_coordinates.size,
        keys=list(active_coordinates.keys),
        entries=selected.entries,
    )
    return RequiredCoordinateSelection(
        coordinates=required_coordinates,
        active_real_rows=selected.active_real_rows,
        value_real_rows=_value_rows_for_entries(
            active_entries=active_coordinates.entries,
            selected_entries=required_coordinates.entries,
            active_real_rows=selected.active_real_rows,
        ),
    )


@dataclass(frozen=True)
class _SelectedRows:
    entries: tuple[DensityEntry, ...]
    active_real_rows: np.ndarray


def _select_required_entries(
    entries: tuple[DensityEntry, ...],
    basis: np.ndarray,
) -> _SelectedRows:
    parameter_count = int(basis.shape[1])
    if parameter_count == 0:
        return _SelectedRows(tuple(), np.empty(0, dtype=int))
    value_count = len(entries)
    _q, r, pivots = la.qr(basis.T, mode="economic", pivoting=True)
    del _q
    tolerance = (
        np.finfo(float).eps * max(basis.shape) * (abs(r[0, 0]) if r.size else 1.0)
    )
    rank = int(np.sum(np.abs(np.diag(r)) > tolerance))
    if rank != parameter_count:
        raise ValueError("Could not select enough active entries to determine params")

    active_real_rows = np.asarray(pivots[:parameter_count], dtype=int)
    selected_positions = sorted(
        {int(pivot % value_count) for pivot in active_real_rows}
    )
    return _SelectedRows(
        tuple(entries[position] for position in selected_positions),
        active_real_rows,
    )


def _coordinates_from_entries(
    *,
    size: int,
    keys: list[tuple[int, ...]],
    entries: tuple[DensityEntry, ...],
) -> DensityCoordinates:
    coordinates = DensityCoordinates.from_entries(
        size=size,
        keys=keys,
        entries=entries,
        allow_empty=True,
    )
    if coordinates is None:  # pragma: no cover - allow_empty guarantees coordinates
        raise ValueError("Required density coordinates unexpectedly missing")
    return coordinates


def _value_rows_for_entries(
    *,
    active_entries: tuple[DensityEntry, ...],
    selected_entries: tuple[DensityEntry, ...],
    active_real_rows: np.ndarray,
) -> np.ndarray:
    value_count = len(active_entries)
    selected_index = {
        entry: position for position, entry in enumerate(selected_entries)
    }
    active_index = {entry: position for position, entry in enumerate(active_entries)}
    active_entries_by_position = {
        position: entry for entry, position in active_index.items()
    }
    selected_count = len(selected_entries)
    rows = []
    for active_row in active_real_rows:
        active_position = int(active_row % value_count)
        entry = active_entries_by_position[active_position]
        selected_position = selected_index[entry]
        if active_row < value_count:
            rows.append(selected_position)
        else:
            rows.append(selected_count + selected_position)
    return np.asarray(rows, dtype=int)
