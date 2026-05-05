from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meanfi.integrate.methods import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    IntegrationMethod,
    UniformGrid,
)
from meanfi.tb.ops import _tb_type, is_sparse_like


def workspace_complex_dtype(integration: IntegrationMethod) -> np.dtype:
    precision = getattr(integration, "workspace_precision", 128)
    if int(precision) == 64:
        return np.dtype(np.complex64)
    if int(precision) == 128:
        return np.dtype(np.complex128)
    raise ValueError("workspace_precision must be 64 or 128")


def workspace_real_dtype(integration: IntegrationMethod) -> np.dtype:
    complex_dtype = workspace_complex_dtype(integration)
    return np.dtype(
        np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
    )


def require_supported_workspace_precision(integration: IntegrationMethod) -> None:
    if (
        isinstance(integration, AdaptiveSimplex)
        and int(integration.workspace_precision) != 128
    ):
        raise ValueError(
            "AdaptiveSimplex currently supports only workspace_precision=128 because the "
            "compiled zero-temperature backend does not yet implement complex64 workspaces"
        )
    if isinstance(integration, (AdaptiveQuadrature, UniformGrid, AdaptiveSimplex)):
        return
    raise TypeError("integration must be an IntegrationMethod instance")


def _structural_entry_indices(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    if is_sparse_like(matrix):
        sparse_coo = matrix.tocoo()
        return np.asarray(sparse_coo.row, dtype=int), np.asarray(
            sparse_coo.col, dtype=int
        )
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
                block[rows, cols] = np.asarray(entry_grid[key], dtype=complex)[
                    start:stop
                ]
            rho[key] = block
        return rho


@dataclass(frozen=True)
class BdGTopHalfSupport:
    normal_support: DensityEntrySupport
    keys: tuple[tuple[int, ...], ...]
    anomalous_rows: tuple[np.ndarray, ...]
    anomalous_cols: tuple[np.ndarray, ...]


def _build_support(
    *,
    size: int,
    keys: list[tuple[int, ...]],
    entries_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
) -> DensityEntrySupport | None:
    return _build_support_allowing_empty(
        size=size,
        keys=keys,
        entries_by_key=entries_by_key,
        allow_empty=False,
    )


def _build_support_allowing_empty(
    *,
    size: int,
    keys: list[tuple[int, ...]],
    entries_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    allow_empty: bool,
) -> DensityEntrySupport | None:
    selected_columns = sorted(
        {
            int(column)
            for rows, cols in entries_by_key.values()
            for column in np.asarray(cols, dtype=int)
        }
    )
    if not selected_columns:
        if not allow_empty:
            return None
        return DensityEntrySupport(
            size=size,
            keys=tuple(keys),
            selected_columns=np.empty(0, dtype=int),
            row_indices=tuple(np.empty(0, dtype=int) for _ in keys),
            col_indices=tuple(np.empty(0, dtype=int) for _ in keys),
            column_positions=tuple(np.empty(0, dtype=int) for _ in keys),
            offsets=tuple(0 for _ in range(len(keys) + 1)),
        )

    lookup = np.full(size, -1, dtype=int)
    lookup[np.asarray(selected_columns, dtype=int)] = np.arange(
        len(selected_columns), dtype=int
    )
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


def _deduplicate_pairs(
    rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.size == 0:
        return rows, cols
    pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _normal_entries_by_key(
    keys: list[tuple[int, ...]],
    interaction_support: _tb_type,
    *,
    ndof: int,
    local_key: tuple[int, ...],
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    diag_support: set[int] = set()
    for interaction in interaction_support.values():
        rows, cols = _structural_entry_indices(interaction)
        diag_support.update(int(row) for row in rows.tolist())
        diag_support.update(int(col) for col in cols.tolist())

    diag = np.asarray(sorted(diag_support), dtype=int)
    entries_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    for key in keys:
        interaction = interaction_support.get(key)
        if interaction is None:
            rows = np.empty(0, dtype=int)
            cols = np.empty(0, dtype=int)
        else:
            rows, cols = _structural_entry_indices(interaction)

        if key == local_key and diag.size:
            rows = np.concatenate([rows, diag])
            cols = np.concatenate([cols, diag])

        rows, cols = _deduplicate_pairs(rows, cols)
        if rows.size or key == local_key:
            entries_by_key[key] = (rows, cols)

    return entries_by_key


def normal_density_entry_support(
    keys: list[tuple[int, ...]],
    interaction_support: _tb_type | None,
    *,
    ndof: int,
    local_key: tuple[int, ...],
    allow_empty: bool = False,
) -> DensityEntrySupport | None:
    if interaction_support is None:
        return None

    entries_by_key = _normal_entries_by_key(
        keys,
        interaction_support,
        ndof=ndof,
        local_key=local_key,
    )
    return _build_support_allowing_empty(
        size=ndof,
        keys=keys,
        entries_by_key=entries_by_key,
        allow_empty=allow_empty,
    )


def full_density_entry_support(
    keys: list[tuple[int, ...]],
    *,
    size: int,
) -> DensityEntrySupport:
    full_rows, full_cols = np.indices((size, size), dtype=int)
    rows = full_rows.reshape(-1)
    cols = full_cols.reshape(-1)
    entries_by_key = {key: (rows, cols) for key in keys}
    support = _build_support_allowing_empty(
        size=size,
        keys=keys,
        entries_by_key=entries_by_key,
        allow_empty=True,
    )
    if support is None:  # pragma: no cover - full support always yields a descriptor
        raise ValueError("Full density support unexpectedly missing")
    return support


def bdg_top_half_support(
    keys: list[tuple[int, ...]],
    interaction_support: _tb_type,
    *,
    ndof: int,
    local_key: tuple[int, ...],
) -> BdGTopHalfSupport:
    normal_support = normal_density_entry_support(
        keys=keys,
        interaction_support=interaction_support,
        ndof=ndof,
        local_key=local_key,
        allow_empty=True,
    )
    if (
        normal_support is None
    ):  # pragma: no cover - allow_empty=True guarantees a descriptor
        raise ValueError("Normal support unexpectedly missing")

    anomalous_rows: list[np.ndarray] = []
    anomalous_cols: list[np.ndarray] = []
    for key in keys:
        interaction = interaction_support.get(key)
        if interaction is None:
            rows = np.empty(0, dtype=int)
            cols = np.empty(0, dtype=int)
        else:
            rows, cols = _structural_entry_indices(interaction)
            rows, cols = _deduplicate_pairs(rows, cols)
        anomalous_rows.append(rows)
        anomalous_cols.append(cols)

    return BdGTopHalfSupport(
        normal_support=normal_support,
        keys=tuple(keys),
        anomalous_rows=tuple(anomalous_rows),
        anomalous_cols=tuple(anomalous_cols),
    )


def bdg_density_entry_support(
    keys: list[tuple[int, ...]],
    interaction_support: _tb_type,
    *,
    ndof: int,
    local_key: tuple[int, ...],
) -> DensityEntrySupport:
    entries_by_key: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    top_half = bdg_top_half_support(
        keys=keys,
        interaction_support=interaction_support,
        ndof=ndof,
        local_key=local_key,
    )
    for index, key in enumerate(keys):
        electron_rows = np.asarray(
            top_half.normal_support.row_indices[index], dtype=int
        )
        electron_cols = np.asarray(
            top_half.normal_support.col_indices[index], dtype=int
        )
        anomalous_rows = np.asarray(top_half.anomalous_rows[index], dtype=int)
        anomalous_cols = np.asarray(top_half.anomalous_cols[index], dtype=int) + ndof
        stacked_rows = np.concatenate([electron_rows, anomalous_rows])
        stacked_cols = np.concatenate([electron_cols, anomalous_cols])
        stacked_rows, stacked_cols = _deduplicate_pairs(stacked_rows, stacked_cols)
        entries_by_key[key] = (stacked_rows, stacked_cols)

    support = _build_support_allowing_empty(
        size=2 * ndof,
        keys=keys,
        entries_by_key=entries_by_key,
        allow_empty=True,
    )
    if support is None:  # pragma: no cover - allow_empty=True guarantees a descriptor
        raise ValueError("BdG density support unexpectedly missing")
    return support
