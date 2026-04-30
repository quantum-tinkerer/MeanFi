from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from meanfi.core.matrix import to_dense
from meanfi.integrate.density_support import BdGTopHalfSupport, DensityEntrySupport
from meanfi.tb.tb import _tb_type


def complex_to_real(z: np.ndarray) -> np.ndarray:
    """Split and concatenate real and imaginary parts of a complex array."""

    array = np.asarray(z, dtype=complex).reshape(-1)
    return np.concatenate((np.real(array), np.imag(array)))


def real_to_complex(z: np.ndarray) -> np.ndarray:
    """Undo `complex_to_real`."""

    array = np.asarray(z, dtype=float).reshape(-1)
    midpoint = len(array) // 2
    return array[:midpoint] + 1j * array[midpoint:]


def _onsite_key(ndim: int) -> tuple[int, ...]:
    return tuple(np.zeros((ndim,), dtype=int))


def _opposite_key(key: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-np.asarray(key, dtype=int))


def _canonical_pair_representative(key: tuple[int, ...]) -> tuple[int, ...]:
    opposite = _opposite_key(key)
    return key if key <= opposite else opposite


def canonical_tb_keys(tb_keys: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return deterministic symmetric TB support with onsite first and pair ordering explicit."""

    keys = [tuple(key) for key in tb_keys]
    if not keys:
        raise ValueError("tb_keys must be non-empty")
    ndim = len(keys[0])
    if any(len(key) != ndim for key in keys):
        raise ValueError("All keys must have the same dimension")

    key_set = set(keys)
    onsite = _onsite_key(ndim)
    if onsite not in key_set:
        raise ValueError("tb_keys must include the onsite key")

    representatives = set()
    for key in key_set:
        opposite = _opposite_key(key)
        if opposite not in key_set:
            raise ValueError("tb_keys must be symmetric under key inversion")
        if key != onsite:
            representatives.add(_canonical_pair_representative(key))

    ordered = [onsite]
    for representative in sorted(representatives):
        ordered.append(representative)
        opposite = _opposite_key(representative)
        if opposite != representative:
            ordered.append(opposite)
    return ordered


def _independent_hopping_keys(tb_keys: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    ordered = canonical_tb_keys(tb_keys)
    onsite = _onsite_key(len(ordered[0]))
    return [
        key
        for key in ordered
        if key != onsite and key == _canonical_pair_representative(key)
    ]


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

    opposite = _opposite_key(key)
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
        onsite_key = _onsite_key(len(ordered_keys[0]))
        onsite_rows, onsite_cols = support_map.get(
            onsite_key,
            (np.empty(0, dtype=int), np.empty(0, dtype=int)),
        )
        onsite_matrix = to_dense(
            tb.get(onsite_key, np.zeros((support.size, support.size), dtype=complex))
        )
        onsite_params = _pack_supported_onsite(
            onsite_matrix,
            onsite_rows,
            onsite_cols,
        )
        hopping_params = []
        for key in _independent_hopping_keys(ordered_keys):
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
    onsite_key = _onsite_key(len(ordered_keys[0]))
    onsite_params = _pack_onsite(tb[onsite_key])
    hopping_params = [
        complex_to_real(np.asarray(tb[key], dtype=complex).reshape(-1))
        for key in _independent_hopping_keys(ordered_keys)
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
        onsite_key = _onsite_key(len(ordered_keys[0]))
        support_map = _support_entries_map(support)
        params = np.asarray(tb_params, dtype=float).reshape(-1)

        onsite_rows, onsite_cols = support_map.get(
            onsite_key,
            (np.empty(0, dtype=int), np.empty(0, dtype=int)),
        )
        onsite, offset = _unpack_supported_onsite(
            params,
            ndof=ndof,
            rows=onsite_rows,
            cols=onsite_cols,
        )

        matrices = {key: np.zeros((ndof, ndof), dtype=complex) for key in ordered_keys}
        matrices[onsite_key] = onsite
        for key in _independent_hopping_keys(ordered_keys):
            rows, cols = _supported_hopping_entries(key, support_map)
            hopping, consumed = _unpack_supported_hopping(
                params[offset:],
                ndof=ndof,
                rows=rows,
                cols=cols,
            )
            offset += consumed
            matrices[key] = hopping
            matrices[_opposite_key(key)] = hopping.conj().T

        if offset != len(params):
            raise ValueError("tb_params has the wrong length for the requested support")
        return matrices

    ordered_keys = canonical_tb_keys(tb_keys)
    params = np.asarray(tb_params, dtype=float).reshape(-1)
    onsite_key = _onsite_key(len(ordered_keys[0]))
    onsite, offset = _unpack_onsite(params, ndof)

    matrices = {onsite_key: onsite}
    block_size = 2 * ndof * ndof
    for key in _independent_hopping_keys(ordered_keys):
        block = real_to_complex(params[offset : offset + block_size]).reshape(ndof, ndof)
        offset += block_size
        matrices[key] = block
        matrices[_opposite_key(key)] = block.conj().T

    if offset != len(params):
        raise ValueError("tb_params has the wrong length for the requested support")
    return matrices


def _split_bdg_matrix(matrix: np.ndarray, ndof: int) -> tuple[np.ndarray, np.ndarray]:
    array = to_dense(matrix)
    return array[:ndof, :ndof], array[:ndof, ndof:]


def _validate_bdg(tb: _tb_type, ndof: int) -> None:
    from meanfi.superconducting.bdg import validate_bdg_tb

    ndim = len(next(iter(tb)))
    validate_bdg_tb(tb, ndof=ndof, ndim=ndim, name="BdG correction")


def bdg_tb_to_rparams(
    tb: _tb_type,
    ndof: int,
    support: BdGTopHalfSupport | None = None,
) -> np.ndarray:
    """Parametrise a BdG correction by independent electron-space normal/anomalous data."""

    _validate_bdg(tb, ndof)
    if support is not None:
        normal_block = {
            key: _split_bdg_matrix(tb.get(key, np.zeros((2 * ndof, 2 * ndof), dtype=complex)), ndof)[0]
            for key in support.keys
        }
        normal_params = tb_to_rparams(normal_block, support=support.normal_support)
        anomalous_parts = []
        for key, rows, cols in zip(
            support.keys,
            support.anomalous_rows,
            support.anomalous_cols,
            strict=True,
        ):
            anomalous = _split_bdg_matrix(
                tb.get(key, np.zeros((2 * ndof, 2 * ndof), dtype=complex)),
                ndof,
            )[1]
            anomalous_parts.append(complex_to_real(np.asarray(anomalous, dtype=complex)[rows, cols]))
        if not anomalous_parts:
            return normal_params
        return np.concatenate((normal_params, *anomalous_parts))

    ordered_keys = canonical_tb_keys(tb.keys())
    normal_block = {}
    anomalous_parts = []
    for key in ordered_keys:
        normal, anomalous = _split_bdg_matrix(tb[key], ndof)
        normal_block[key] = normal
        anomalous_parts.append(complex_to_real(anomalous.reshape(-1)))

    normal_params = tb_to_rparams(normal_block)
    return np.concatenate((normal_params, *anomalous_parts))


def rparams_to_bdg_tb(
    tb_params: np.ndarray,
    tb_keys: list[tuple[None] | tuple[int, ...]],
    ndof: int,
    support: BdGTopHalfSupport | None = None,
) -> _tb_type:
    """Reconstruct a BdG correction from independent electron-space normal/anomalous data."""

    if support is not None:
        normal_size = tb_to_rparams(
            {key: np.zeros((ndof, ndof), dtype=complex) for key in support.normal_support.keys},
            support=support.normal_support,
        ).size
        params = np.asarray(tb_params, dtype=float).reshape(-1)
        normal_block = rparams_to_tb(
            params[:normal_size],
            list(support.normal_support.keys),
            ndof,
            support=support.normal_support,
        )

        offset = normal_size
        anomalous_block = {key: np.zeros((ndof, ndof), dtype=complex) for key in support.keys}
        for key, rows, cols in zip(
            support.keys,
            support.anomalous_rows,
            support.anomalous_cols,
            strict=True,
        ):
            count = int(rows.size)
            if count:
                values = real_to_complex(params[offset : offset + 2 * count])
                anomalous_block[key][rows, cols] = values
                offset += 2 * count
        if offset != len(params):
            raise ValueError("tb_params has the wrong length for the requested BdG support")

        from meanfi.superconducting.bdg import assemble_bdg_correction
        from types import SimpleNamespace

        tb = assemble_bdg_correction(
            normal_block,
            anomalous_block,
            SimpleNamespace(_ndof=ndof),
        )
        _validate_bdg(tb, ndof)
        return tb

    ordered_keys = canonical_tb_keys(tb_keys)
    n_onsite = ndof + ndof * (ndof - 1)
    n_hopping = len(_independent_hopping_keys(ordered_keys)) * 2 * ndof * ndof
    normal_size = n_onsite + n_hopping

    params = np.asarray(tb_params, dtype=float).reshape(-1)
    normal_block = rparams_to_tb(params[:normal_size], ordered_keys, ndof)

    block_size = 2 * ndof * ndof
    offset = normal_size
    anomalous_block = {}
    for key in ordered_keys:
        anomalous = real_to_complex(params[offset : offset + block_size]).reshape(ndof, ndof)
        anomalous_block[key] = anomalous
        offset += block_size

    if offset != len(params):
        raise ValueError("tb_params has the wrong length for the requested BdG support")

    from meanfi.superconducting.bdg import assemble_bdg_correction
    from types import SimpleNamespace

    model = SimpleNamespace(_ndof=ndof)
    tb = assemble_bdg_correction(normal_block, anomalous_block, model)
    _validate_bdg(tb, ndof)
    return tb


def bdg_density_to_rparams(
    density_matrix: _tb_type,
    *,
    support: BdGTopHalfSupport,
    ndof: int,
) -> np.ndarray:
    electron_density = {
        key: np.asarray(density_matrix.get(key, np.zeros((2 * ndof, 2 * ndof), dtype=complex)))[:ndof, :ndof]
        for key in support.keys
    }
    normal_params = tb_to_rparams(electron_density, support=support.normal_support)
    anomalous_parts = []
    for key, rows, cols in zip(
        support.keys,
        support.anomalous_rows,
        support.anomalous_cols,
        strict=True,
    ):
        anomalous = np.asarray(
            density_matrix.get(key, np.zeros((2 * ndof, 2 * ndof), dtype=complex)),
            dtype=complex,
        )[:ndof, ndof:]
        anomalous_parts.append(complex_to_real(anomalous[rows, cols]))
    if not anomalous_parts:
        return normal_params
    return np.concatenate((normal_params, *anomalous_parts))


def rparams_to_bdg_density(
    params: np.ndarray,
    *,
    support: BdGTopHalfSupport,
    ndof: int,
) -> _tb_type:
    normal_size = tb_to_rparams(
        {key: np.zeros((ndof, ndof), dtype=complex) for key in support.normal_support.keys},
        support=support.normal_support,
    ).size
    electron_density = rparams_to_tb(
        np.asarray(params, dtype=float)[:normal_size],
        list(support.normal_support.keys),
        ndof,
        support=support.normal_support,
    )
    offset = normal_size
    density: _tb_type = {}
    for key in support.keys:
        density[key] = np.zeros((2 * ndof, 2 * ndof), dtype=complex)
        density[key][:ndof, :ndof] = electron_density.get(key, np.zeros((ndof, ndof), dtype=complex))

    for key, rows, cols in zip(
        support.keys,
        support.anomalous_rows,
        support.anomalous_cols,
        strict=True,
    ):
        count = int(rows.size)
        if count:
            values = real_to_complex(np.asarray(params, dtype=float)[offset : offset + 2 * count])
            density[key][:ndof, ndof:][rows, cols] = values
            offset += 2 * count

    if offset != len(np.asarray(params, dtype=float).reshape(-1)):
        raise ValueError("tb_params has the wrong length for the requested BdG density support")
    return density
