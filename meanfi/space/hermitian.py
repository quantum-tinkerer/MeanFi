from __future__ import annotations

import numpy as np

from meanfi.space.density_selection import (
    DensitySelection,
    selected_pairs_by_key,
    sorted_unique_pairs,
)
from meanfi.space.params import (
    canonical_tb_keys,
    complex_to_real,
    independent_hopping_keys,
    opposite_key,
    onsite_key,
    real_to_complex,
)
from meanfi.tb.ops import _tb_type, to_dense


def _hermitian_onsite_matrix_to_real_params(matrix: np.ndarray) -> np.ndarray:
    onsite = np.asarray(matrix, dtype=complex)
    upper = onsite[np.triu_indices(onsite.shape[0], k=1)]
    diagonal = np.diag(onsite).real
    return np.concatenate((diagonal, complex_to_real(upper)))


def _real_params_to_hermitian_onsite_matrix(
    params: np.ndarray,
    ndof: int,
) -> tuple[np.ndarray, int]:
    diagonal = np.asarray(params[:ndof], dtype=float)
    n_upper = ndof * (ndof - 1) // 2
    upper = real_to_complex(params[ndof : ndof + 2 * n_upper])

    onsite = np.zeros((ndof, ndof), dtype=complex)
    onsite[np.diag_indices(ndof)] = diagonal
    onsite[np.triu_indices(ndof, k=1)] = upper
    onsite += np.triu(onsite, k=1).conj().T
    return onsite, ndof + 2 * n_upper


def _independent_onsite_entries_from_selection(
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
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
    return diagonal, upper_pairs


def _selected_onsite_entries_to_real_params(
    matrix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    diagonal, upper_pairs = _independent_onsite_entries_from_selection(rows, cols)
    diagonal_params = np.real(np.asarray(matrix, dtype=complex)[diagonal, diagonal])
    upper_params = np.asarray(
        [np.asarray(matrix, dtype=complex)[row, col] for row, col in upper_pairs],
        dtype=complex,
    )
    if upper_params.size == 0:
        return diagonal_params.astype(float, copy=False)
    return np.concatenate(
        [diagonal_params.astype(float, copy=False), complex_to_real(upper_params)]
    )


def _real_params_to_selected_onsite_matrix(
    params: np.ndarray,
    *,
    ndof: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, int]:
    diagonal, upper_pairs = _independent_onsite_entries_from_selection(rows, cols)
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


def _fold_selected_hopping_entries_to_representative_key(
    key: tuple[int, ...],
    selected_pairs: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    cols = []
    if key in selected_pairs:
        rows.append(np.asarray(selected_pairs[key][0], dtype=int))
        cols.append(np.asarray(selected_pairs[key][1], dtype=int))

    opposite = opposite_key(key)
    if opposite in selected_pairs:
        opp_rows = np.asarray(selected_pairs[opposite][0], dtype=int)
        opp_cols = np.asarray(selected_pairs[opposite][1], dtype=int)
        rows.append(opp_cols)
        cols.append(opp_rows)

    if not rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    selected_rows = np.concatenate(rows)
    selected_cols = np.concatenate(cols)
    if selected_rows.size == 0:
        return selected_rows, selected_cols
    return sorted_unique_pairs(selected_rows, selected_cols)


def _selected_hopping_entries_to_real_params(
    matrix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    if rows.size == 0:
        return np.empty(0, dtype=float)
    values = np.asarray(matrix, dtype=complex)[rows, cols]
    return complex_to_real(values)


def _real_params_to_selected_hopping_matrix(
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


def full_hermitian_tb_to_real_params(tb: _tb_type) -> np.ndarray:
    """Convert a full Hermitian TB dictionary into minimal real parameters."""

    ordered_keys = canonical_tb_keys(tb.keys())
    local_key = onsite_key(len(ordered_keys[0]))
    onsite_params = _hermitian_onsite_matrix_to_real_params(tb[local_key])
    hopping_params = [
        complex_to_real(np.asarray(tb[key], dtype=complex).reshape(-1))
        for key in independent_hopping_keys(ordered_keys)
    ]
    if not hopping_params:
        return onsite_params
    return np.concatenate((onsite_params, *hopping_params))


def selected_hermitian_tb_to_real_params(
    tb: _tb_type,
    selection: DensitySelection,
) -> np.ndarray:
    """Convert selected Hermitian TB entries into minimal real parameters."""

    selected_pairs = selected_pairs_by_key(selection)
    ordered_keys = canonical_tb_keys(selection.keys)
    local_key = onsite_key(len(ordered_keys[0]))
    onsite_rows, onsite_cols = selected_pairs.get(
        local_key,
        (np.empty(0, dtype=int), np.empty(0, dtype=int)),
    )
    onsite_matrix = to_dense(
        tb.get(local_key, np.zeros((selection.size, selection.size), dtype=complex))
    )
    onsite_params = _selected_onsite_entries_to_real_params(
        onsite_matrix,
        onsite_rows,
        onsite_cols,
    )
    hopping_params = []
    for key in independent_hopping_keys(ordered_keys):
        rows, cols = _fold_selected_hopping_entries_to_representative_key(
            key,
            selected_pairs,
        )
        if rows.size == 0:
            continue
        hopping_params.append(
            _selected_hopping_entries_to_real_params(
                to_dense(
                    tb.get(
                        key,
                        np.zeros((selection.size, selection.size), dtype=complex),
                    )
                ),
                rows,
                cols,
            )
        )
    if not hopping_params:
        return onsite_params
    return np.concatenate((onsite_params, *hopping_params))


def hermitian_param_count(selection: DensitySelection, ndof: int) -> int:
    """Return the real-coordinate length for a selected Hermitian TB space."""

    zero_tb = {key: np.zeros((ndof, ndof), dtype=complex) for key in selection.keys}
    return int(selected_hermitian_tb_to_real_params(zero_tb, selection).size)


def real_params_to_selected_hermitian_tb(
    tb_params: np.ndarray,
    selection: DensitySelection,
    ndof: int,
) -> _tb_type:
    """Convert selected Hermitian real parameters into a TB dictionary."""

    ordered_keys = canonical_tb_keys(selection.keys)
    local_key = onsite_key(len(ordered_keys[0]))
    selected_pairs = selected_pairs_by_key(selection)
    params = np.asarray(tb_params, dtype=float).reshape(-1)

    onsite_rows, onsite_cols = selected_pairs.get(
        local_key,
        (np.empty(0, dtype=int), np.empty(0, dtype=int)),
    )
    onsite, offset = _real_params_to_selected_onsite_matrix(
        params,
        ndof=ndof,
        rows=onsite_rows,
        cols=onsite_cols,
    )

    matrices = {key: np.zeros((ndof, ndof), dtype=complex) for key in ordered_keys}
    matrices[local_key] = onsite
    for key in independent_hopping_keys(ordered_keys):
        rows, cols = _fold_selected_hopping_entries_to_representative_key(
            key,
            selected_pairs,
        )
        hopping, consumed = _real_params_to_selected_hopping_matrix(
            params[offset:],
            ndof=ndof,
            rows=rows,
            cols=cols,
        )
        offset += consumed
        matrices[key] = hopping
        matrices[opposite_key(key)] = hopping.conj().T

    if offset != len(params):
        raise ValueError("tb_params has the wrong length for the requested selection")
    return matrices


def real_params_to_full_hermitian_tb(
    tb_params: np.ndarray,
    tb_keys: list[tuple[None] | tuple[int, ...]],
    ndof: int,
) -> _tb_type:
    """Convert full Hermitian real parameters into a TB dictionary."""

    ordered_keys = canonical_tb_keys(tb_keys)
    params = np.asarray(tb_params, dtype=float).reshape(-1)
    local_key = onsite_key(len(ordered_keys[0]))
    onsite, offset = _real_params_to_hermitian_onsite_matrix(params, ndof)

    matrices = {local_key: onsite}
    block_size = 2 * ndof * ndof
    for key in independent_hopping_keys(ordered_keys):
        block = real_to_complex(params[offset : offset + block_size]).reshape(
            ndof, ndof
        )
        offset += block_size
        matrices[key] = block
        matrices[opposite_key(key)] = block.conj().T

    if offset != len(params):
        raise ValueError("tb_params has the wrong length for the requested selection")
    return matrices
