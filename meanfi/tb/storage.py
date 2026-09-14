from __future__ import annotations

from meanfi.tb.ops import _tb_type, is_sparse_like
from meanfi.tb.validate import matrix_allclose


def prefers_sparse_storage(*tb_dicts: _tb_type) -> bool:
    return any(is_sparse_like(matrix) for tb in tb_dicts for matrix in tb.values())


def tb_entries_changed(
    original: _tb_type,
    projected: _tb_type,
    *,
    atol: float = 1e-12,
) -> bool:
    for key in original.keys() | projected.keys():
        before, after = original.get(key), projected.get(key)
        if before is None:
            before = after * 0
        if after is None:
            after = before * 0
        if not matrix_allclose(before, after, atol=atol):
            return True
    return False
