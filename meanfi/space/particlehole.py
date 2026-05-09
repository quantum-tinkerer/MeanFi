from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.space.density_selection import (
    DensitySelection,
    density_selection_from_pairs,
    matrix_support_pairs,
    sorted_unique_pairs,
)
from meanfi.space.hermitian import normal_density_selection
from meanfi.tb.ops import _tb_type
from meanfi.tb.ops import to_dense


@dataclass(frozen=True)
class BdGTopHalfSelection:
    """Selected electron and anomalous values from the top half of BdG blocks."""

    electron: DensitySelection
    anomalous: DensitySelection


def split_bdg_matrix(matrix: np.ndarray, ndof: int) -> tuple[np.ndarray, np.ndarray]:
    """Return electron and anomalous top-half blocks of a BdG matrix."""

    array = to_dense(matrix)
    return array[:ndof, :ndof], array[:ndof, ndof:]


def bdg_top_half_selection(
    *,
    keys: list[tuple[int, ...]],
    interaction_tb: _tb_type,
    ndof: int,
    local_key: tuple[int, ...],
) -> BdGTopHalfSelection:
    electron_selection = normal_density_selection(
        keys=keys,
        interaction_tb=interaction_tb,
        ndof=ndof,
        local_key=local_key,
        allow_empty=True,
    )
    if electron_selection is None:  # pragma: no cover - allow_empty guarantees selection
        raise ValueError("BdG top-half selection unexpectedly missing")

    anomalous_pairs: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    for key in keys:
        matrix = interaction_tb.get(key)
        rows, cols = (
            matrix_support_pairs(matrix)
            if matrix is not None
            else (np.empty(0, dtype=int), np.empty(0, dtype=int))
        )
        anomalous_pairs[key] = sorted_unique_pairs(rows, cols)
    anomalous_selection = density_selection_from_pairs(
        size=ndof,
        keys=keys,
        selected_pairs=anomalous_pairs,
        allow_empty=True,
    )
    if anomalous_selection is None:  # pragma: no cover - allow_empty guarantees selection
        raise ValueError("BdG anomalous selection unexpectedly missing")

    return BdGTopHalfSelection(
        electron=electron_selection,
        anomalous=anomalous_selection,
    )


def bdg_density_selection_from_top_half(
    top_half_selection: BdGTopHalfSelection,
    *,
    ndof: int,
) -> DensitySelection:
    """Select top-half BdG density values needed by BdG mean-field updates."""

    electron_by_key = {
        key_selection.key: key_selection
        for key_selection in top_half_selection.electron.key_selections
    }
    anomalous_by_key = {
        key_selection.key: key_selection
        for key_selection in top_half_selection.anomalous.key_selections
    }
    selected_pairs: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    for key in top_half_selection.electron.keys:
        electron = electron_by_key.get(key)
        anomalous = anomalous_by_key.get(key)
        rows = []
        cols = []
        if electron is not None and electron.rows.size:
            rows.append(electron.rows)
            cols.append(electron.cols)
        if anomalous is not None and anomalous.rows.size:
            rows.append(anomalous.rows)
            cols.append(anomalous.cols + ndof)
        if rows:
            selected_pairs[key] = sorted_unique_pairs(
                np.concatenate(rows),
                np.concatenate(cols),
            )
        else:
            selected_pairs[key] = (np.empty(0, dtype=int), np.empty(0, dtype=int))

    selection = density_selection_from_pairs(
        size=2 * ndof,
        keys=list(top_half_selection.electron.keys),
        selected_pairs=selected_pairs,
        allow_empty=True,
    )
    if selection is None:  # pragma: no cover - allow_empty guarantees selection
        raise ValueError("BdG density selection unexpectedly missing")
    return selection


def bdg_density_selection(
    *,
    keys: list[tuple[int, ...]],
    interaction_tb: _tb_type,
    ndof: int,
    local_key: tuple[int, ...],
) -> DensitySelection:
    top_half_selection = bdg_top_half_selection(
        keys=keys,
        interaction_tb=interaction_tb,
        ndof=ndof,
        local_key=local_key,
    )
    return bdg_density_selection_from_top_half(top_half_selection, ndof=ndof)
