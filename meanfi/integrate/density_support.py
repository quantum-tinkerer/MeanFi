from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meanfi.core.matrix import is_sparse_like
from meanfi.integrate.methods import AdaptiveQuadrature, AdaptiveSimplex, IntegrationMethod, UniformGrid
from meanfi.tb.tb import _tb_type


def workspace_complex_dtype(integration: IntegrationMethod) -> np.dtype:
    precision = getattr(integration, "workspace_precision", 128)
    if int(precision) == 64:
        return np.dtype(np.complex64)
    if int(precision) == 128:
        return np.dtype(np.complex128)
    raise ValueError("workspace_precision must be 64 or 128")


def workspace_real_dtype(integration: IntegrationMethod) -> np.dtype:
    complex_dtype = workspace_complex_dtype(integration)
    return np.dtype(np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64)


def require_supported_workspace_precision(integration: IntegrationMethod) -> None:
    if isinstance(integration, AdaptiveSimplex) and int(integration.workspace_precision) != 128:
        raise ValueError(
            "AdaptiveSimplex currently supports only workspace_precision=128 because the "
            "compiled zero-temperature backend does not yet implement complex64 workspaces"
        )
    if isinstance(integration, (AdaptiveQuadrature, UniformGrid, AdaptiveSimplex)):
        return
    raise TypeError("integration must be an IntegrationMethod instance")


def _structural_entry_indices(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    if is_sparse_like(matrix):
        coo = matrix.tocoo()
        return np.asarray(coo.row, dtype=int), np.asarray(coo.col, dtype=int)
    rows, cols = np.nonzero(np.asarray(matrix))
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


@dataclass(frozen=True)
class DensityEntrySupport:
    size: int
    keys: tuple[tuple[int, ...], ...]
    selected_columns: np.ndarray
    row_indices: tuple[np.ndarray, ...]
    col_indices: tuple[np.ndarray, ...]
    column_positions: tuple[np.ndarray, ...]
    offsets: tuple[int, ...]

    @property
    def output_size(self) -> int:
        return int(self.offsets[-1]) if self.offsets else 0

    def basis_block(self, *, dtype: np.dtype) -> np.ndarray:
        block = np.zeros((self.size, self.selected_columns.size), dtype=dtype)
        block[self.selected_columns, np.arange(self.selected_columns.size)] = 1.0
        return block

    def pack_columns(
        self,
        density_columns: np.ndarray,
        *,
        phases: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.empty(self.output_size, dtype=density_columns.dtype)
        for index, (rows, positions) in enumerate(
            zip(self.row_indices, self.column_positions, strict=True)
        ):
            start = self.offsets[index]
            stop = self.offsets[index + 1]
            if stop == start:
                continue
            entry_values = density_columns[rows, positions]
            if phases is not None:
                entry_values = entry_values * phases[index]
            values[start:stop] = entry_values
        return values

    def expand_entries(
        self,
        estimate: np.ndarray,
        error: np.ndarray,
    ) -> tuple[_tb_type, _tb_type]:
        estimate = np.asarray(estimate)
        error = np.asarray(error)
        rho: _tb_type = {}
        rho_error: _tb_type = {}
        error_dtype = error.dtype if error.size else float
        for index, key in enumerate(self.keys):
            block = np.zeros((self.size, self.size), dtype=complex)
            error_block = np.zeros((self.size, self.size), dtype=error_dtype)
            start = self.offsets[index]
            stop = self.offsets[index + 1]
            if stop > start:
                rows = self.row_indices[index]
                cols = self.col_indices[index]
                block[rows, cols] = estimate[start:stop]
                error_block[rows, cols] = error[start:stop]
            rho[key] = block
            rho_error[key] = error_block
        return rho, rho_error

    def expand_ifft_entries(self, entry_grid: np.ndarray) -> _tb_type:
        rho: _tb_type = {}
        for index, key in enumerate(self.keys):
            block = np.zeros((self.size, self.size), dtype=complex)
            start = self.offsets[index]
            stop = self.offsets[index + 1]
            if stop > start:
                rows = self.row_indices[index]
                cols = self.col_indices[index]
                block[rows, cols] = np.asarray(entry_grid[key], dtype=complex)[start:stop]
            rho[key] = block
        return rho


def _build_support(
    *,
    size: int,
    keys: list[tuple[int, ...]],
    entries_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
) -> DensityEntrySupport | None:
    selected_columns = sorted(
        {
            int(column)
            for rows, cols in entries_by_key.values()
            for column in np.asarray(cols, dtype=int)
        }
    )
    if not selected_columns:
        return None

    lookup = np.full(size, -1, dtype=int)
    lookup[np.asarray(selected_columns, dtype=int)] = np.arange(len(selected_columns), dtype=int)
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    column_positions: list[np.ndarray] = []
    offsets = [0]
    for key in keys:
        rows, cols = entries_by_key.get(
            key,
            (np.empty(0, dtype=int), np.empty(0, dtype=int)),
        )
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        row_indices.append(rows)
        col_indices.append(cols)
        column_positions.append(lookup[cols] if cols.size else np.empty(0, dtype=int))
        offsets.append(offsets[-1] + int(rows.size))

    return DensityEntrySupport(
        size=size,
        keys=tuple(keys),
        selected_columns=np.asarray(selected_columns, dtype=int),
        row_indices=tuple(row_indices),
        col_indices=tuple(col_indices),
        column_positions=tuple(column_positions),
        offsets=tuple(offsets),
    )


def normal_density_entry_support(
    keys: list[tuple[int, ...]],
    interaction_support: _tb_type | None,
    *,
    ndof: int,
    local_key: tuple[int, ...],
) -> DensityEntrySupport | None:
    if interaction_support is None:
        return None

    entries_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    diag = np.arange(ndof, dtype=int)
    entries_by_key[local_key] = (diag, diag)
    for key in keys:
        interaction = interaction_support.get(key)
        if interaction is None:
            continue
        rows, cols = _structural_entry_indices(interaction)
        if rows.size == 0 and key != local_key:
            continue
        if key == local_key and rows.size:
            rows = np.concatenate([diag, rows])
            cols = np.concatenate([diag, cols])
            pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
            rows, cols = pairs[:, 0], pairs[:, 1]
        entries_by_key[key] = (rows, cols)

    return _build_support(size=ndof, keys=keys, entries_by_key=entries_by_key)


def bdg_density_entry_support(
    keys: list[tuple[int, ...]],
    interaction_support: _tb_type,
    *,
    ndof: int,
    local_key: tuple[int, ...],
) -> DensityEntrySupport:
    entries_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    diag = np.arange(ndof, dtype=int)
    entries_by_key[local_key] = (diag, diag)

    for key in keys:
        interaction = interaction_support.get(key)
        if interaction is None:
            continue
        rows, cols = _structural_entry_indices(interaction)
        if rows.size == 0 and key != local_key:
            continue

        electron_rows = rows
        electron_cols = cols
        anomalous_rows = rows
        anomalous_cols = cols + ndof
        stacked_rows = np.concatenate([electron_rows, anomalous_rows])
        stacked_cols = np.concatenate([electron_cols, anomalous_cols])
        if key == local_key:
            stacked_rows = np.concatenate([diag, stacked_rows])
            stacked_cols = np.concatenate([diag, stacked_cols])
        pairs = np.unique(np.stack([stacked_rows, stacked_cols], axis=1), axis=0)
        entries_by_key[key] = (pairs[:, 0], pairs[:, 1])

    support = _build_support(size=2 * ndof, keys=keys, entries_by_key=entries_by_key)
    if support is None:
        raise ValueError("BdG density support unexpectedly had no selected entries")
    return support
