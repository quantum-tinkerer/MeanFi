from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.space.coordinates import DensityCoordinates, DensityEntry


@dataclass(frozen=True)
class RequiredCoordinateSelection:
    coordinates: DensityCoordinates
    real_rows: np.ndarray


def select_required_coordinates(
    active_coordinates: DensityCoordinates,
    basis: np.ndarray,
) -> RequiredCoordinateSelection:
    """Choose active entries whose real/imag values determine all parameters."""

    selected_entries = _select_required_entries(active_coordinates.entries, basis)
    required_coordinates = _coordinates_from_entries(
        size=active_coordinates.size,
        keys=list(active_coordinates.keys),
        entries=selected_entries,
    )
    return RequiredCoordinateSelection(
        coordinates=required_coordinates,
        real_rows=_real_rows_for_entries(
            active_entries=active_coordinates.entries,
            selected_entries=required_coordinates.entries,
        ),
    )


def _select_required_entries(
    entries: tuple[DensityEntry, ...],
    basis: np.ndarray,
) -> tuple[DensityEntry, ...]:
    parameter_count = int(basis.shape[1])
    if parameter_count == 0:
        return tuple()
    value_count = len(entries)
    selected_rows: list[int] = []
    selected_entries: list[DensityEntry] = []
    rank = 0
    for index, entry in enumerate(entries):
        candidate_rows = [*selected_rows, index, value_count + index]
        candidate_rank = int(np.linalg.matrix_rank(basis[candidate_rows, :]))
        if candidate_rank <= rank:
            continue
        selected_rows = candidate_rows
        selected_entries.append(entry)
        rank = candidate_rank
        if rank == parameter_count:
            break
    if rank != parameter_count:
        raise ValueError("Could not select enough active entries to determine params")
    return tuple(selected_entries)


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


def _real_rows_for_entries(
    *,
    active_entries: tuple[DensityEntry, ...],
    selected_entries: tuple[DensityEntry, ...],
) -> np.ndarray:
    value_count = len(active_entries)
    index = {entry: position for position, entry in enumerate(active_entries)}
    positions = [index[entry] for entry in selected_entries]
    rows = [*positions, *(value_count + position for position in positions)]
    return np.asarray(rows, dtype=int)
