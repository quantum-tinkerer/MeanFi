from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from meanfi.meanfield import bdg_density_keys
from meanfi.space.coordinates import (
    DensityCoordinates,
    canonical_tb_keys,
    matrix_support_pairs,
    onsite_key,
    sorted_unique_pairs,
)
from meanfi.tb.ops import _tb_type

if TYPE_CHECKING:
    from meanfi.model import Model


@dataclass(frozen=True)
class ActiveCoordinateSupport:
    coordinates: DensityCoordinates
    interaction_keys: list[tuple[int, ...]]
    density_keys: list[tuple[int, ...]]
    onsite: tuple[int, ...]


def active_tb_keys(keys) -> list[tuple[int, ...]]:
    key_set = {tuple(key) for key in keys}
    if not key_set:
        key_set.add(tuple())
    key_set.add(onsite_key(len(next(iter(key_set)))))
    return canonical_tb_keys(key_set)


def normal_active_support(model: Model) -> ActiveCoordinateSupport:
    onsite = model._local_key
    interaction_keys = list(model.h_int)
    density_keys = active_tb_keys([*interaction_keys, onsite])
    coordinates = _coordinates_from_pairs(
        size=model._ndof,
        keys=density_keys,
        pairs=_normal_active_pairs_from_interaction(
            model.h_int,
            keys=density_keys,
            onsite=onsite,
        ),
        label="Normal",
    )
    return ActiveCoordinateSupport(
        coordinates=coordinates,
        interaction_keys=interaction_keys,
        density_keys=density_keys,
        onsite=onsite,
    )


def bdg_active_support(model: Model) -> ActiveCoordinateSupport:
    onsite = onsite_key(model._ndim)
    density_keys = active_tb_keys(bdg_density_keys(model, {}))
    electron_pairs = _normal_active_pairs_from_interaction(
        model.h_int,
        keys=density_keys,
        onsite=onsite,
    )
    anomalous_pairs = _bdg_anomalous_pairs_from_interaction(
        model.h_int,
        keys=density_keys,
        ndof=model._ndof,
    )
    coordinates = _coordinates_from_pairs(
        size=2 * model._ndof,
        keys=density_keys,
        pairs=_merge_pair_maps(electron_pairs, anomalous_pairs),
        label="BdG",
    )
    return ActiveCoordinateSupport(
        coordinates=coordinates,
        interaction_keys=list(model.h_int),
        density_keys=density_keys,
        onsite=onsite,
    )


def _normal_active_pairs_from_interaction(
    h_int: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    onsite: tuple[int, ...],
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    """Read density entries that can affect the normal mean-field map."""

    pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {key: [] for key in keys}
    diagonal_indices: set[int] = set()
    for matrix in h_int.values():
        rows, cols = matrix_support_pairs(matrix)
        diagonal_indices.update(int(row) for row in rows)
        diagonal_indices.update(int(col) for col in cols)

    for key in keys:
        matrix = h_int.get(key)
        if matrix is not None:
            rows, cols = matrix_support_pairs(matrix)
            _append_pairs(pairs, key, rows, cols)
        if key == onsite:
            for index in diagonal_indices:
                pairs[key].append((index, index))
    return _materialize_pairs(pairs)


def _bdg_anomalous_pairs_from_interaction(
    h_int: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    ndof: int,
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {key: [] for key in keys}
    for key in keys:
        matrix = h_int.get(key)
        if matrix is None:
            continue
        rows, cols = matrix_support_pairs(matrix)
        shifted_cols = ndof + cols
        _append_pairs(pairs, key, rows, shifted_cols)
    return _materialize_pairs(pairs)


def _merge_pair_maps(
    *pair_maps: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    merged: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for pair_map in pair_maps:
        for key, (rows, cols) in pair_map.items():
            _append_pairs(merged, key, rows, cols)
    return _materialize_pairs(merged)


def _coordinates_from_pairs(
    *,
    size: int,
    keys: list[tuple[int, ...]],
    pairs: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    label: str,
) -> DensityCoordinates:
    coordinates = DensityCoordinates.from_pairs(
        size=size,
        keys=keys,
        pairs_by_key=pairs,
        allow_empty=True,
    )
    if coordinates is None:  # pragma: no cover - allow_empty guarantees coordinates
        raise ValueError(f"{label} active coordinates unexpectedly missing")
    return coordinates


def _append_pairs(
    pairs: dict[tuple[int, ...], list[tuple[int, int]]],
    key: tuple[int, ...],
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    pairs.setdefault(key, [])
    for row, col in zip(rows, cols, strict=True):
        pairs[key].append((int(row), int(col)))


def _materialize_pairs(
    pairs: dict[tuple[int, ...], list[tuple[int, int]]],
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    result = {}
    for key, entries in pairs.items():
        if entries:
            rows, cols = np.asarray(entries, dtype=int).T
            result[key] = sorted_unique_pairs(rows, cols)
        else:
            result[key] = (np.empty(0, dtype=int), np.empty(0, dtype=int))
    return result
