from __future__ import annotations

from collections.abc import Mapping
from copy import copy

from meanfi.tb.ops import _tb_type, is_sparse_like
from meanfi.tb.validate import matrix_allclose


class _MatrixView(Mapping):
    """Expose isolated matrix containers sharing the model's read-only arrays.

    Sparse mutators can replace their arrays, bypassing NumPy's read-only flag.
    A shallow container copy keeps such replacements outside the owned model.
    """

    def __init__(self, blocks):
        self._blocks = blocks

    def __getitem__(self, key):
        block = self._blocks[key]
        return copy(block) if is_sparse_like(block) else block.view()

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)


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
