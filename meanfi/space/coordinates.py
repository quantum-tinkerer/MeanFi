from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix

from meanfi.tb.ops import _tb_type, is_sparse_like

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
        canonical_pair_representative(key) for key in key_set if key != local_key
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
    value_slices: tuple[slice, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, Integral)
            or self.size <= 0
        ):
            raise ValueError("density coordinate size must be a positive integer")
        count = len(self.keys)
        if len(set(self.keys)) != count:
            raise ValueError("density coordinate keys must be unique")
        if len(self.rows_by_key) != count or len(self.cols_by_key) != count:
            raise ValueError("density coordinate arrays must match the key count")
        rows_by_key = []
        cols_by_key = []
        value_slices = []
        offset = 0
        for rows, cols in zip(self.rows_by_key, self.cols_by_key, strict=True):
            for values in (rows, cols):
                array = np.asarray(values)
                if array.size and array.dtype.kind not in "iu":
                    raise ValueError("density row/column coordinates must be integers")
            rows = np.array(rows, dtype=int, copy=True)
            cols = np.array(cols, dtype=int, copy=True)
            if rows.ndim != 1 or cols.ndim != 1 or rows.size != cols.size:
                raise ValueError(
                    "density row/column coordinates must be paired vectors"
                )
            if np.any(rows < 0) or np.any(rows >= self.size):
                raise ValueError("density row coordinate is out of bounds")
            if np.any(cols < 0) or np.any(cols >= self.size):
                raise ValueError("density column coordinate is out of bounds")
            indices = np.sort(rows * self.size + cols)
            if np.any(indices[1:] == indices[:-1]):
                raise ValueError("density coordinates must be unique within each key")
            value_slices.append(slice(offset, offset + rows.size))
            rows.setflags(write=False)
            cols.setflags(write=False)
            rows_by_key.append(rows)
            cols_by_key.append(cols)
            offset += rows.size
        object.__setattr__(self, "rows_by_key", tuple(rows_by_key))
        object.__setattr__(self, "cols_by_key", tuple(cols_by_key))
        object.__setattr__(self, "value_slices", tuple(value_slices))

    @property
    def is_full(self) -> bool:
        """Whether every matrix entry is present for every listed key."""

        # Bounds and uniqueness are checked when the layout is constructed.
        return all(rows.size == self.size * self.size for rows in self.rows_by_key)

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
            matches = np.flatnonzero((rows == row) & (cols == col))
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

    def values_from_tb(self, tb: _tb_type) -> np.ndarray:
        """Pack TB dictionary entries into this coordinate order."""

        values = np.empty(self.value_count, dtype=complex)
        for key, rows, cols, value_slice in self.iter_key_coordinates():
            block = tb.get(key)
            if not rows.size:
                continue
            if block is None:
                raise ValueError(f"density is missing required coordinate key {key}")
            if block.shape != (self.size, self.size):
                raise ValueError("density coordinate matrix sizes do not match")
            if is_sparse_like(block):
                selected = np.asarray(block.tocsr()[rows, cols]).reshape(-1)
            else:
                selected = np.asarray(block)[rows, cols]
            values[value_slice] = selected
        return values

    @classmethod
    def from_pairs(
        cls,
        *,
        size: int,
        keys: list[tuple[int, ...]],
        pairs_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    ) -> DensityCoordinates:
        rows_by_key: list[np.ndarray] = []
        cols_by_key: list[np.ndarray] = []
        for key in keys:
            rows, cols = pairs_by_key.get(
                key,
                (np.empty(0, dtype=int), np.empty(0, dtype=int)),
            )
            rows_by_key.append(rows)
            cols_by_key.append(cols)
        return cls(
            size=size,
            keys=tuple(keys),
            rows_by_key=tuple(rows_by_key),
            cols_by_key=tuple(cols_by_key),
        )

    @classmethod
    def from_entries(
        cls,
        *,
        size: int,
        keys: list[tuple[int, ...]],
        entries: tuple[DensityEntry, ...],
    ) -> DensityCoordinates:
        pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {key: [] for key in keys}
        for key, row, col in entries:
            if key not in pairs:
                raise ValueError("density entry key is absent from the coordinate keys")
            if any(
                isinstance(index, bool) or not isinstance(index, Integral)
                for index in (row, col)
            ):
                raise ValueError("density row/column coordinates must be integers")
            pairs[key].append((row, col))
        return cls.from_pairs(
            size=size,
            keys=keys,
            pairs_by_key=_materialize_entry_pairs(pairs),
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

    pairs = {}
    if keys:
        grid = np.arange(size, dtype=int)
        rows, cols = np.meshgrid(grid, grid, indexing="ij")
        pairs = {key: (rows.reshape(-1), cols.reshape(-1)) for key in keys}
    return DensityCoordinates.from_pairs(
        size=size,
        keys=keys,
        pairs_by_key=pairs,
    )


def _assemble_blocks(
    coordinates: DensityCoordinates, values: np.ndarray, *, sparse: bool = False
) -> _tb_type:
    """Place coordinate values into dense blocks, or CSR blocks when requested."""

    values = np.asarray(values)
    if values.ndim != 1 or values.size != coordinates.value_count:
        raise ValueError("values must match the density coordinate count")
    rho: _tb_type = {}
    for key, rows, cols, value_slice in coordinates.iter_key_coordinates():
        if sparse:
            block = csr_matrix(
                (values[value_slice], (rows, cols)),
                shape=(coordinates.size, coordinates.size),
                dtype=complex,
            )
        else:
            block = np.zeros((coordinates.size, coordinates.size), dtype=complex)
            block[rows, cols] = values[value_slice]
        rho[key] = block
    return rho
