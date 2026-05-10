from __future__ import annotations

import numpy as np

from meanfi.space.density_selection import (
    DensitySelection,
    density_selection_from_pairs,
    matrix_support_pairs,
    sorted_unique_pairs,
)
from meanfi.tb.ops import _tb_type


def _density_pairs_needed_by_normal_interaction(
    *,
    keys: list[tuple[int, ...]],
    interaction_tb: _tb_type,
    local_key: tuple[int, ...],
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    """Select density entries touched by a normal mean-field interaction."""

    selected_pairs: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    diagonal_indices: set[int] = set()
    for matrix in interaction_tb.values():
        rows, cols = matrix_support_pairs(matrix)
        diagonal_indices.update(int(row) for row in rows)
        diagonal_indices.update(int(col) for col in cols)

    for key in keys:
        matrix = interaction_tb.get(key)
        rows, cols = (
            matrix_support_pairs(matrix)
            if matrix is not None
            else (np.empty(0, dtype=int), np.empty(0, dtype=int))
        )
        if key == local_key and diagonal_indices:
            diagonal = np.asarray(sorted(diagonal_indices), dtype=int)
            rows = np.concatenate([rows, diagonal])
            cols = np.concatenate([cols, diagonal])
        rows, cols = sorted_unique_pairs(rows, cols)
        if rows.size or key == local_key:
            selected_pairs[key] = (rows, cols)

    if local_key not in selected_pairs:
        selected_pairs[local_key] = (np.empty(0, dtype=int), np.empty(0, dtype=int))
    return selected_pairs


def normal_density_selection(
    *,
    keys: list[tuple[int, ...]],
    interaction_tb: _tb_type,
    ndof: int,
    local_key: tuple[int, ...],
    allow_empty: bool = False,
) -> DensitySelection | None:
    """Build the selected normal density values required by an interaction TB."""

    return density_selection_from_pairs(
        size=ndof,
        keys=keys,
        selected_pairs=_density_pairs_needed_by_normal_interaction(
            keys=keys,
            interaction_tb=interaction_tb,
            local_key=local_key,
        ),
        allow_empty=allow_empty,
    )
