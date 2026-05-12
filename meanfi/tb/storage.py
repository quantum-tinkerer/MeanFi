from __future__ import annotations

import numpy as np

from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like, to_dense


def prefers_sparse_storage(*tb_dicts: _tb_type) -> bool:
    return any(
        is_sparse_like(matrix)
        for tb in tb_dicts
        if tb is not None
        for matrix in tb.values()
    )


def match_tb_storage(tb: _tb_type, *, like_sparse: bool) -> _tb_type:
    if not like_sparse:
        return tb
    return {key: as_sparse(value) for key, value in tb.items()}


def tb_entries_changed(
    original: _tb_type,
    projected: _tb_type,
    *,
    atol: float = 1e-12,
) -> bool:
    for key in frozenset(original) | frozenset(projected):
        before = to_dense(original.get(key, np.zeros((0, 0), dtype=complex)))
        after = to_dense(projected.get(key, np.zeros((0, 0), dtype=complex)))
        if before.shape != after.shape:
            continue
        if np.any(np.abs(before - after) > atol):
            return True
    return False
