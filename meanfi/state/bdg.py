from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from meanfi.physics.bdg import assemble_bdg_correction, validate_bdg_tb
from meanfi.state.keys import (
    canonical_tb_keys,
    complex_to_real,
    independent_hopping_keys,
    real_to_complex,
)
from meanfi.state.normal import rparams_to_tb, tb_to_rparams
from meanfi.state.support import BdGTopHalfSupport
from meanfi.tb.ops import _tb_type, to_dense


def _split_bdg_matrix(matrix: np.ndarray, ndof: int) -> tuple[np.ndarray, np.ndarray]:
    array = to_dense(matrix)
    return array[:ndof, :ndof], array[:ndof, ndof:]


def _validate_bdg(tb: _tb_type, ndof: int) -> None:
    ndim = len(next(iter(tb)))
    validate_bdg_tb(tb, ndof=ndof, ndim=ndim, name="BdG correction")


def bdg_tb_to_rparams(
    tb: _tb_type,
    ndof: int,
    support: BdGTopHalfSupport | None = None,
) -> np.ndarray:
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

        tb = assemble_bdg_correction(
            normal_block,
            anomalous_block,
            SimpleNamespace(_ndof=ndof),
        )
        _validate_bdg(tb, ndof)
        return tb

    ordered_keys = canonical_tb_keys(tb_keys)
    n_onsite = ndof + ndof * (ndof - 1)
    n_hopping = len(independent_hopping_keys(ordered_keys)) * 2 * ndof * ndof
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

    tb = assemble_bdg_correction(
        normal_block,
        anomalous_block,
        SimpleNamespace(_ndof=ndof),
    )
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

    params = np.asarray(params, dtype=float).reshape(-1)
    for key, rows, cols in zip(
        support.keys,
        support.anomalous_rows,
        support.anomalous_cols,
        strict=True,
    ):
        count = int(rows.size)
        if count:
            values = real_to_complex(params[offset : offset + 2 * count])
            density[key][:ndof, ndof:][rows, cols] = values
            offset += 2 * count

    if offset != len(params):
        raise ValueError("tb_params has the wrong length for the requested BdG density support")
    return density
