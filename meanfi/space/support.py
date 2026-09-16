"""Density entries used by the normal and anomalous interaction maps."""

from meanfi.space.coordinates import (
    DensityCoordinates,
    canonical_tb_keys,
    matrix_support_pairs,
    onsite_key,
)
from meanfi.tb.ops import _tb_type
from meanfi.tb.validate import tb_dimension, tb_orbital_count


def active_tb_keys(keys) -> list[tuple[int, ...]]:
    key_set = {tuple(key) for key in keys} or {()}
    key_set.add(onsite_key(len(next(iter(key_set)))))
    return canonical_tb_keys(key_set)


def _active_support(h_int: _tb_type, *, superconducting: bool) -> DensityCoordinates:
    size = tb_orbital_count(h_int)
    onsite = onsite_key(tb_dimension(h_int))
    entries = set()
    for key, matrix in h_int.items():
        rows, cols = matrix_support_pairs(matrix)
        for row, col in zip(rows, cols, strict=True):
            entries.update(((onsite, row, row), (onsite, col, col), (key, row, col)))
            if superconducting:
                entries.add((key, row, size + col))
    return DensityCoordinates.from_entries(
        size=(2 if superconducting else 1) * size,
        keys=active_tb_keys(h_int),
        entries=entries,
    )


def normal_active_support(h_int: _tb_type) -> DensityCoordinates:
    return _active_support(h_int, superconducting=False)


def bdg_active_support(h_int: _tb_type) -> DensityCoordinates:
    return _active_support(h_int, superconducting=True)
