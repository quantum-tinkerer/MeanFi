from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sparse
from scipy.linalg import block_diag as scipy_block_diag

_tb_type = dict[tuple[int, ...], np.ndarray]


def is_sparse_like(matrix: Any) -> bool:
    return sparse.issparse(matrix)


def to_dense(matrix: Any) -> np.ndarray:
    if is_sparse_like(matrix):
        return np.asarray(matrix.toarray(), dtype=complex)
    return np.asarray(matrix, dtype=complex)


def as_sparse(matrix: Any):
    if is_sparse_like(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix, dtype=complex))


def matrix_shape(matrix: Any) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("Tight-binding values must be matrices")
    return int(shape[0]), int(shape[1])


def transpose(matrix: Any):
    return matrix.T


def conjugate_transpose(matrix: Any):
    return matrix.conj().T


def elementwise_product(lhs: Any, rhs: Any):
    if is_sparse_like(lhs):
        return lhs.multiply(np.asarray(rhs, dtype=complex)).tocsr()
    if is_sparse_like(rhs):
        return rhs.multiply(np.asarray(lhs, dtype=complex)).tocsr()
    return np.asarray(lhs, dtype=complex) * np.asarray(rhs, dtype=complex)


def block_diag(top: Any, bottom: Any):
    if is_sparse_like(top) or is_sparse_like(bottom):
        return sparse.block_diag((as_sparse(top), as_sparse(bottom)), format="csr")

    return scipy_block_diag(
        np.asarray(top, dtype=complex),
        np.asarray(bottom, dtype=complex),
    )


def matrix_bound(matrix: Any) -> float:
    if is_sparse_like(matrix):
        row_sums = np.asarray(abs(matrix).sum(axis=1)).ravel()
        return float(np.max(row_sums)) if row_sums.size else 0.0
    array = np.asarray(matrix)
    if array.size == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(array), axis=1)))


def add_tb(tb1: _tb_type, tb2: _tb_type) -> _tb_type:
    return {
        key: tb1.get(key, 0) + tb2.get(key, 0)
        for key in frozenset(tb1) | frozenset(tb2)
    }


def scale_tb(tb: _tb_type, scale: float) -> _tb_type:
    return {key: tb.get(key, 0) * scale for key in frozenset(tb)}


def compare_dicts(dict1: dict, dict2: dict, atol: float = 1e-10) -> None:
    for key in frozenset(dict1) | frozenset(dict2):
        assert np.allclose(dict1[key], dict2[key], atol=atol)
