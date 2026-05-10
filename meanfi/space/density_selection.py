from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.tb.ops import _tb_type, is_sparse_like, to_dense


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
    """Return lexicographically sorted unique row/col pairs.

    ``np.unique(..., axis=0)`` both deduplicates and sorts; selected-value order
    therefore becomes deterministic here.
    """

    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.size == 0:
        return rows, cols
    pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


@dataclass(frozen=True)
class DensityKeySelection:
    """Selected density values belonging to one tight-binding key."""

    key: tuple[int, ...]
    rows: np.ndarray
    cols: np.ndarray
    value_slice: slice


@dataclass(frozen=True)
class DensitySelection:
    """Ordered rho_R[row, col] values needed by a mean-field calculation."""

    size: int
    key_selections: tuple[DensityKeySelection, ...]

    @property
    def keys(self) -> tuple[tuple[int, ...], ...]:
        return tuple(selection.key for selection in self.key_selections)

    @property
    def value_count(self) -> int:
        if not self.key_selections:
            return 0
        return int(self.key_selections[-1].value_slice.stop or 0)

    @property
    def all_rows(self) -> np.ndarray:
        arrays = [
            selection.rows for selection in self.key_selections if selection.rows.size
        ]
        if not arrays:
            return np.empty(0, dtype=int)
        return np.concatenate(arrays).astype(int, copy=False)

    @property
    def all_cols(self) -> np.ndarray:
        arrays = [
            selection.cols for selection in self.key_selections if selection.cols.size
        ]
        if not arrays:
            return np.empty(0, dtype=int)
        return np.concatenate(arrays).astype(int, copy=False)

    def key_slice(self, key: tuple[int, ...]) -> slice:
        for selection in self.key_selections:
            if selection.key == key:
                return selection.value_slice
        raise KeyError(key)

    def values_from_assembled_matrix(
        self,
        matrix: np.ndarray,
        *,
        phases: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample selected values from one assembled k-space density matrix.

        The same matrix is sampled for each tight-binding key. Optional phases
        multiply each key's selected values before they enter the flat value
        vector.
        """

        matrix = np.asarray(matrix)
        values = np.empty(self.value_count, dtype=matrix.dtype)
        for index, selection in enumerate(self.key_selections):
            selected = matrix[selection.rows, selection.cols]
            if phases is not None:
                selected = selected * phases[index]
            values[selection.value_slice] = selected
        return values

    def phase_values(self, values: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """Multiply a selected-value vector by one phase per key selection."""

        phased = np.array(values, copy=True)
        for index, selection in enumerate(self.key_selections):
            phased[selection.value_slice] *= phases[index]
        return phased

    def values_from_tb(self, tb: _tb_type) -> np.ndarray:
        """Pack selected TB dictionary values into the selection's flat order."""

        values = np.empty(self.value_count, dtype=complex)
        for selection in self.key_selections:
            block = tb.get(selection.key)
            if block is None:
                selected = np.zeros(selection.rows.size, dtype=complex)
            else:
                selected = to_dense(block)[selection.rows, selection.cols]
            values[selection.value_slice] = selected
        return values

    def values_to_tb(self, values: np.ndarray) -> _tb_type:
        """Expand selected values into sparse-in-value dense TB blocks."""

        values = np.asarray(values)
        rho: _tb_type = {}
        for selection in self.key_selections:
            block = np.zeros((self.size, self.size), dtype=complex)
            if selection.rows.size:
                block[selection.rows, selection.cols] = values[selection.value_slice]
            rho[selection.key] = block
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
        for selection in self.key_selections:
            block = np.zeros((self.size, self.size), dtype=error_dtype)
            if selection.rows.size:
                block[selection.rows, selection.cols] = errors[selection.value_slice]
            rho_error[selection.key] = block
        return rho, rho_error


def selected_pairs_by_key(
    selection: DensitySelection,
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    return {
        key_selection.key: (
            np.asarray(key_selection.rows, dtype=int),
            np.asarray(key_selection.cols, dtype=int),
        )
        for key_selection in selection.key_selections
    }


def density_selection_from_pairs(
    *,
    size: int,
    keys: list[tuple[int, ...]],
    selected_pairs: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    allow_empty: bool,
) -> DensitySelection | None:
    key_selections: list[DensityKeySelection] = []
    offset = 0
    has_values = False
    for key in keys:
        rows, cols = selected_pairs.get(
            key,
            (np.empty(0, dtype=int), np.empty(0, dtype=int)),
        )
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        count = int(rows.size)
        has_values = has_values or count > 0
        key_selections.append(
            DensityKeySelection(
                key=key,
                rows=rows,
                cols=cols,
                value_slice=slice(offset, offset + count),
            )
        )
        offset += count

    if not has_values and not allow_empty:
        return None
    return DensitySelection(size=size, key_selections=tuple(key_selections))


def full_density_selection(
    keys: list[tuple[int, ...]],
    *,
    size: int,
) -> DensitySelection:
    """Select every matrix entry for each requested tight-binding key."""

    grid = np.arange(size, dtype=int)
    rows, cols = np.meshgrid(grid, grid, indexing="ij")
    pairs = {key: (rows.reshape(-1), cols.reshape(-1)) for key in keys}
    selection = density_selection_from_pairs(
        size=size,
        keys=keys,
        selected_pairs=pairs,
        allow_empty=False,
    )
    if selection is None:  # pragma: no cover - dense full selection cannot be empty
        raise ValueError("Full density selection unexpectedly empty")
    return selection
