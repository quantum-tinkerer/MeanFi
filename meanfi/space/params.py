from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def complex_to_real(z: np.ndarray) -> np.ndarray:
    """Pack complex values as ``[real parts..., imaginary parts...]``."""

    array = np.asarray(z, dtype=complex).reshape(-1)
    return np.concatenate((np.real(array), np.imag(array)))


def real_to_complex(z: np.ndarray) -> np.ndarray:
    """Unpack ``[real parts..., imaginary parts...]`` into complex values."""

    array = np.asarray(z, dtype=float).reshape(-1)
    if len(array) % 2:
        raise ValueError("real_to_complex expects an even number of real values")
    midpoint = len(array) // 2
    return array[:midpoint] + 1j * array[midpoint:]


def onsite_key(ndim: int) -> tuple[int, ...]:
    return (0,) * ndim


def opposite_key(key: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-component for component in key)


def canonical_pair_representative(key: tuple[int, ...]) -> tuple[int, ...]:
    """Choose the deterministic representative from the Hermitian pair {R, -R}."""

    opposite = opposite_key(key)
    return key if key <= opposite else opposite


def canonical_tb_keys(tb_keys: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return onsite first, then deterministic {R, -R} key pairs."""

    keys = [tuple(key) for key in tb_keys]
    if not keys:
        raise ValueError("tb_keys must be non-empty")
    ndim = len(keys[0])
    if any(len(key) != ndim for key in keys):
        raise ValueError("All keys must have the same dimension")

    key_set = set(keys)
    local_key = onsite_key(ndim)
    if local_key not in key_set:
        raise ValueError("tb_keys must include the onsite key")

    representatives = set()
    for key in key_set:
        opposite = opposite_key(key)
        if opposite not in key_set:
            raise ValueError("tb_keys must be symmetric under key inversion")
        if key != local_key:
            representatives.add(canonical_pair_representative(key))

    ordered = [local_key]
    for representative in sorted(representatives):
        ordered.append(representative)
        opposite = opposite_key(representative)
        if opposite != representative:
            ordered.append(opposite)
    return ordered


def independent_hopping_keys(
    tb_keys: Iterable[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Return one hopping key from each non-onsite Hermitian pair {R, -R}."""

    ordered = canonical_tb_keys(tb_keys)
    local_key = onsite_key(len(ordered[0]))
    return [
        key
        for key in ordered
        if key != local_key and key == canonical_pair_representative(key)
    ]
