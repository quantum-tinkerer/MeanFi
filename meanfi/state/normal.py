from __future__ import annotations

import numpy as np

from meanfi.state.keys import (
    canonical_tb_keys,
    complex_to_real,
    independent_hopping_keys,
    opposite_key,
    onsite_key,
    real_to_complex,
)
from meanfi.state.support import DensityEntrySupport
from meanfi.tb.ops import _tb_type, to_dense


def _pack_onsite(matrix: np.ndarray) -> np.ndarray:
    onsite = np.asarray(matrix, dtype=complex)
    upper = onsite[np.triu_indices(onsite.shape[0], k=1)]
    diagonal = np.diag(onsite).real
    return np.concatenate((diagonal, complex_to_real(upper)))


def _unpack_onsite(params: np.ndarray, ndof: int) -> tuple[np.ndarray, int]:
    diagonal = np.asarray(params[:ndof], dtype=float)
    n_upper = ndof * (ndof - 1) // 2
    upper = real_to_complex(params[ndof : ndof + 2 * n_upper])

    onsite = np.zeros((ndof, ndof), dtype=complex)
    onsite[np.diag_indices(ndof)] = diagonal
    onsite[np.triu_indices(ndof, k=1)] = upper
    onsite += np.triu(onsite, k=1).conj().T
    return onsite, ndof + 2 * n_upper


def _support_entries_map(
    support: DensityEntrySupport,
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    return {
        key: (
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
        )
        for key, rows, cols in zip(
            support.keys,
            support.row_indices,
            support.col_indices,
            strict=True,
        )
    }


def _pack_supported_onsite(
    matrix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    diagonal = np.asarray(
        sorted({int(row) for row, col in zip(rows, cols, strict=True) if row == col}),
        dtype=int,
    )
    upper_pairs = sorted(
        {
            (min(int(row), int(col)), max(int(row), int(col)))
            for row, col in zip(rows, cols, strict=True)
            if row != col
        }
    )
    diagonal_params = np.real(np.asarray(matrix, dtype=complex)[diagonal, diagonal])
    upper_params = np.asarray(
        [np.asarray(matrix, dtype=complex)[row, col] for row, col in upper_pairs],
        dtype=complex,
    )
    if upper_params.size == 0:
        return diagonal_params.astype(float, copy=False)
    return np.concatenate([diagonal_params.astype(float, copy=False), complex_to_real(upper_params)])


def _unpack_supported_onsite(
    params: np.ndarray,
    *,
    ndof: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, int]:
    diagonal = np.asarray(
        sorted({int(row) for row, col in zip(rows, cols, strict=True) if row == col}),
        dtype=int,
    )
    upper_pairs = sorted(
        {
            (min(int(row), int(col)), max(int(row), int(col)))
            for row, col in zip(rows, cols, strict=True)
            if row != col
        }
    )
    onsite = np.zeros((ndof, ndof), dtype=complex)
    offset = 0
    if diagonal.size:
        onsite[diagonal, diagonal] = np.asarray(params[: diagonal.size], dtype=float)
        offset += int(diagonal.size)
    if upper_pairs:
        upper_values = real_to_complex(params[offset : offset + 2 * len(upper_pairs)])
        offset += 2 * len(upper_pairs)
        for (row, col), value in zip(upper_pairs, upper_values, strict=True):
            onsite[row, col] = value
            onsite[col, row] = np.conjugate(value)
    return onsite, offset


def _supported_hopping_entries(
    key: tuple[int, ...],
    support_map: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    cols = []
    if key in support_map:
        rows.append(np.asarray(support_map[key][0], dtype=int))
        cols.append(np.asarray(support_map[key][1], dtype=int))

    opposite = opposite_key(key)
    if opposite in support_map:
        opp_rows = np.asarray(support_map[opposite][0], dtype=int)
        opp_cols = np.asarray(support_map[opposite][1], dtype=int)
        rows.append(opp_cols)
        cols.append(opp_rows)

    if not rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    supported_rows = np.concatenate(rows)
    supported_cols = np.concatenate(cols)
    if supported_rows.size == 0:
        return supported_rows, supported_cols
    pairs = np.unique(np.stack([supported_rows, supported_cols], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _pack_supported_hopping(
    matrix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    if rows.size == 0:
        return np.empty(0, dtype=float)
    values = np.asarray(matrix, dtype=complex)[rows, cols]
    return complex_to_real(values)


def _unpack_supported_hopping(
    params: np.ndarray,
    *,
    ndof: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, int]:
    hopping = np.zeros((ndof, ndof), dtype=complex)
    if rows.size == 0:
        return hopping, 0
    count = int(rows.size)
    values = real_to_complex(params[: 2 * count])
    hopping[rows, cols] = values
    return hopping, 2 * count


def tb_to_rparams(
    tb: _tb_type,
    support: DensityEntrySupport | None = None,
) -> np.ndarray:
    """Parametrise a Hermitian tight-binding dictionary by a real vector."""

    if support is not None:
        support_map = _support_entries_map(support)
        ordered_keys = canonical_tb_keys(support.keys)
        local_key = onsite_key(len(ordered_keys[0]))
        onsite_rows, onsite_cols = support_map.get(
            local_key,
            (np.empty(0, dtype=int), np.empty(0, dtype=int)),
        )
        onsite_matrix = to_dense(
            tb.get(local_key, np.zeros((support.size, support.size), dtype=complex))
        )
        onsite_params = _pack_supported_onsite(
            onsite_matrix,
            onsite_rows,
            onsite_cols,
        )
        hopping_params = []
        for key in independent_hopping_keys(ordered_keys):
            rows, cols = _supported_hopping_entries(key, support_map)
            if rows.size == 0:
                continue
            hopping_params.append(
                _pack_supported_hopping(
                    to_dense(
                        tb.get(key, np.zeros((support.size, support.size), dtype=complex))
                    ),
                    rows,
                    cols,
                )
            )
        if not hopping_params:
            return onsite_params
        return np.concatenate((onsite_params, *hopping_params))

    ordered_keys = canonical_tb_keys(tb.keys())
    local_key = onsite_key(len(ordered_keys[0]))
    onsite_params = _pack_onsite(tb[local_key])
    hopping_params = [
        complex_to_real(np.asarray(tb[key], dtype=complex).reshape(-1))
        for key in independent_hopping_keys(ordered_keys)
    ]
    if not hopping_params:
        return onsite_params
    return np.concatenate((onsite_params, *hopping_params))


def rparams_to_tb(
    tb_params: np.ndarray,
    tb_keys: list[tuple[None] | tuple[int, ...]],
    ndof: int,
    support: DensityEntrySupport | None = None,
) -> _tb_type:
    """Extract a Hermitian tight-binding dictionary from a real vector parametrisation."""

    if support is not None:
        ordered_keys = canonical_tb_keys(support.keys)
        local_key = onsite_key(len(ordered_keys[0]))
        support_map = _support_entries_map(support)
        params = np.asarray(tb_params, dtype=float).reshape(-1)

        onsite_rows, onsite_cols = support_map.get(
            local_key,
            (np.empty(0, dtype=int), np.empty(0, dtype=int)),
        )
        onsite, offset = _unpack_supported_onsite(
            params,
            ndof=ndof,
            rows=onsite_rows,
            cols=onsite_cols,
        )

        matrices = {key: np.zeros((ndof, ndof), dtype=complex) for key in ordered_keys}
        matrices[local_key] = onsite
        for key in independent_hopping_keys(ordered_keys):
            rows, cols = _supported_hopping_entries(key, support_map)
            hopping, consumed = _unpack_supported_hopping(
                params[offset:],
                ndof=ndof,
                rows=rows,
                cols=cols,
            )
            offset += consumed
            matrices[key] = hopping
            matrices[opposite_key(key)] = hopping.conj().T

        if offset != len(params):
            raise ValueError("tb_params has the wrong length for the requested support")
        return matrices

    ordered_keys = canonical_tb_keys(tb_keys)
    params = np.asarray(tb_params, dtype=float).reshape(-1)
    local_key = onsite_key(len(ordered_keys[0]))
    onsite, offset = _unpack_onsite(params, ndof)

    matrices = {local_key: onsite}
    block_size = 2 * ndof * ndof
    for key in independent_hopping_keys(ordered_keys):
        block = real_to_complex(params[offset : offset + block_size]).reshape(ndof, ndof)
        offset += block_size
        matrices[key] = block
        matrices[opposite_key(key)] = block.conj().T

    if offset != len(params):
        raise ValueError("tb_params has the wrong length for the requested support")
    return matrices
