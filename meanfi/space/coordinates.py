from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from meanfi.tb.ops import _tb_type, is_sparse_like, to_dense

DensityEntry = tuple[tuple[int, ...], int, int]


def onsite_key(ndim: int) -> tuple[int, ...]:
    return (0,) * ndim


def opposite_key(key: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-component for component in key)


def canonical_pair_representative(key: tuple[int, ...]) -> tuple[int, ...]:
    """Choose the deterministic representative from the pair {R, -R}."""

    opposite = opposite_key(key)
    return key if key <= opposite else opposite


def canonical_tb_keys(keys) -> list[tuple[int, ...]]:
    """Return onsite first, then deterministic {R, -R} key pairs."""

    normalized = [tuple(key) for key in keys]
    if not normalized:
        raise ValueError("tb_keys must be non-empty")
    ndim = len(normalized[0])
    if any(len(key) != ndim for key in normalized):
        raise ValueError("All keys must have the same dimension")
    key_set = set(normalized)
    local_key = onsite_key(ndim)
    if local_key not in key_set:
        raise ValueError("tb_keys must include the onsite key")

    representatives = {
        canonical_pair_representative(key)
        for key in key_set
        if key != local_key
    }
    for key in key_set:
        if opposite_key(key) not in key_set:
            raise ValueError("tb_keys must be symmetric under key inversion")

    ordered = [local_key]
    for representative in sorted(representatives):
        ordered.append(representative)
        opposite = opposite_key(representative)
        if opposite != representative:
            ordered.append(opposite)
    return ordered


def matrix_support_pairs(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return row/col positions touched by a matrix.

    Sparse matrices use their stored sparsity pattern. Dense matrices use their
    nonzero entries.
    """

    if is_sparse_like(matrix):
        coordinate_matrix = matrix.tocoo()
        return (
            coordinate_matrix.row.astype(int, copy=False),
            coordinate_matrix.col.astype(int, copy=False),
        )
    rows, cols = np.nonzero(np.asarray(matrix))
    return rows.astype(int, copy=False), cols.astype(int, copy=False)


def sorted_unique_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lexicographically sorted unique row/col pairs."""

    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.size == 0:
        return rows, cols
    pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


@dataclass(frozen=True)
class DensityCoordinates:
    """Ordered complex density entries ``rho_R[row, col]``.

    This class is only a coordinate layout. It does not know why entries were
    selected, which symmetries they obey, or how they become solver parameters.
    """

    size: int
    keys: tuple[tuple[int, ...], ...]
    rows_by_key: tuple[np.ndarray, ...]
    cols_by_key: tuple[np.ndarray, ...]
    value_slices: tuple[slice, ...]

    @property
    def value_count(self) -> int:
        if not self.value_slices:
            return 0
        return int(self.value_slices[-1].stop or 0)

    @property
    def all_rows(self) -> np.ndarray:
        arrays = [rows for rows in self.rows_by_key if rows.size]
        if not arrays:
            return np.empty(0, dtype=int)
        return np.concatenate(arrays).astype(int, copy=False)

    @property
    def all_cols(self) -> np.ndarray:
        arrays = [cols for cols in self.cols_by_key if cols.size]
        if not arrays:
            return np.empty(0, dtype=int)
        return np.concatenate(arrays).astype(int, copy=False)

    def iter_key_coordinates(
        self,
    ) -> Iterator[tuple[tuple[int, ...], np.ndarray, np.ndarray, slice]]:
        for key, rows, cols, value_slice in zip(
            self.keys,
            self.rows_by_key,
            self.cols_by_key,
            self.value_slices,
            strict=True,
        ):
            yield key, rows, cols, value_slice

    @property
    def entries(self) -> tuple[DensityEntry, ...]:
        return tuple(
            (key, int(row), int(col))
            for key, rows, cols, _value_slice in self.iter_key_coordinates()
            for row, col in zip(rows, cols, strict=True)
        )

    def key_slice(self, key: tuple[int, ...]) -> slice:
        for candidate, value_slice in zip(self.keys, self.value_slices, strict=True):
            if candidate == key:
                return value_slice
        raise KeyError(key)

    def index(self, key: tuple[int, ...], row: int, col: int) -> int:
        for candidate, rows, cols, value_slice in self.iter_key_coordinates():
            if candidate != key:
                continue
            matches = np.flatnonzero((rows == int(row)) & (cols == int(col)))
            if matches.size:
                return int((value_slice.start or 0) + matches[0])
        raise KeyError((key, row, col))

    def values_from_assembled_matrix(
        self,
        matrix: np.ndarray,
        *,
        phases: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample selected entries from one assembled k-space density matrix."""

        matrix = np.asarray(matrix)
        values = np.empty(self.value_count, dtype=matrix.dtype)
        for index, (_key, rows, cols, value_slice) in enumerate(
            self.iter_key_coordinates()
        ):
            selected = matrix[rows, cols]
            if phases is not None:
                selected = selected * phases[index]
            values[value_slice] = selected
        return values

    def phase_values(self, values: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """Multiply a coordinate value vector by one phase per key."""

        phased = np.array(values, copy=True)
        for index, (_key, _rows, _cols, value_slice) in enumerate(
            self.iter_key_coordinates()
        ):
            phased[value_slice] *= phases[index]
        return phased

    def values_from_tb(self, tb: _tb_type) -> np.ndarray:
        """Pack TB dictionary entries into this coordinate order."""

        values = np.empty(self.value_count, dtype=complex)
        for key, rows, cols, value_slice in self.iter_key_coordinates():
            block = tb.get(key)
            if block is None:
                selected = np.zeros(rows.size, dtype=complex)
            else:
                selected = to_dense(block)[rows, cols]
            values[value_slice] = selected
        return values

    def values_to_tb(self, values: np.ndarray) -> _tb_type:
        """Expand coordinate values into sparse-in-value dense TB blocks."""

        values = np.asarray(values)
        rho: _tb_type = {}
        for key, rows, cols, value_slice in self.iter_key_coordinates():
            block = np.zeros((self.size, self.size), dtype=complex)
            if rows.size:
                block[rows, cols] = values[value_slice]
            rho[key] = block
        return rho

    def values_and_errors_to_tb(
        self,
        values: np.ndarray,
        errors: np.ndarray,
    ) -> tuple[_tb_type, _tb_type]:
        values = np.asarray(values)
        errors = np.asarray(errors)
        rho = self.values_to_tb(values)
        rho_error: _tb_type = {}
        error_dtype = errors.dtype if errors.size else float
        for key, rows, cols, value_slice in self.iter_key_coordinates():
            block = np.zeros((self.size, self.size), dtype=error_dtype)
            if rows.size:
                block[rows, cols] = errors[value_slice]
            rho_error[key] = block
        return rho, rho_error

    @classmethod
    def from_pairs(
        cls,
        *,
        size: int,
        keys: list[tuple[int, ...]],
        pairs_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
        allow_empty: bool,
    ) -> DensityCoordinates | None:
        rows_by_key: list[np.ndarray] = []
        cols_by_key: list[np.ndarray] = []
        value_slices: list[slice] = []
        offset = 0
        has_values = False
        for key in keys:
            rows, cols = pairs_by_key.get(
                key,
                (np.empty(0, dtype=int), np.empty(0, dtype=int)),
            )
            rows = np.asarray(rows, dtype=int)
            cols = np.asarray(cols, dtype=int)
            if rows.size != cols.size:
                raise ValueError("coordinate rows and cols must have matching size")
            count = int(rows.size)
            has_values = has_values or count > 0
            rows_by_key.append(rows)
            cols_by_key.append(cols)
            value_slices.append(slice(offset, offset + count))
            offset += count
        if not has_values and not allow_empty:
            return None
        return cls(
            size=int(size),
            keys=tuple(keys),
            rows_by_key=tuple(rows_by_key),
            cols_by_key=tuple(cols_by_key),
            value_slices=tuple(value_slices),
        )

    @classmethod
    def from_entries(
        cls,
        *,
        size: int,
        keys: list[tuple[int, ...]],
        entries: tuple[DensityEntry, ...],
        allow_empty: bool,
    ) -> DensityCoordinates | None:
        pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {key: [] for key in keys}
        for key, row, col in entries:
            pairs.setdefault(key, []).append((int(row), int(col)))
        return cls.from_pairs(
            size=size,
            keys=keys,
            pairs_by_key=_materialize_entry_pairs(pairs),
            allow_empty=allow_empty,
        )


def _materialize_entry_pairs(
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


def full_density_coordinates(
    keys: list[tuple[int, ...]],
    *,
    size: int,
) -> DensityCoordinates:
    """Select every matrix entry for each requested tight-binding key."""

    grid = np.arange(size, dtype=int)
    rows, cols = np.meshgrid(grid, grid, indexing="ij")
    pairs = {key: (rows.reshape(-1), cols.reshape(-1)) for key in keys}
    coords = DensityCoordinates.from_pairs(
        size=size,
        keys=keys,
        pairs_by_key=pairs,
        allow_empty=False,
    )
    if coords is None:  # pragma: no cover - full coordinates cannot be empty
        raise ValueError("Full density coordinates unexpectedly empty")
    return coords
