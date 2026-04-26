from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from meanfi.core.matrix import to_dense
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


def tb_to_rparams(tb: _tb_type) -> np.ndarray:
    """Parametrise a Hermitian tight-binding dictionary by a real vector."""

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
) -> _tb_type:
    """Extract a Hermitian tight-binding dictionary from a real vector parametrisation."""

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


def bdg_tb_to_rparams(tb: _tb_type, ndof: int) -> np.ndarray:
    """Parametrise a BdG correction by independent electron-space normal/anomalous data."""

    _validate_bdg(tb, ndof)
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
) -> _tb_type:
    """Reconstruct a BdG correction from independent electron-space normal/anomalous data."""

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
