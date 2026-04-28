from __future__ import annotations

from typing import Any

import numpy as np


def is_sparse_like(matrix: Any) -> bool:
    return hasattr(matrix, "toarray") and hasattr(matrix, "tocsr")


def sparse_module():
    try:
        import scipy.sparse as sparse
    except ImportError as exc:  # pragma: no cover - depends on optional scipy
        raise ImportError("Sparse mean-field inputs require scipy to be installed") from exc
    return sparse


def sparse_linalg_module():
    try:
        import scipy.sparse.linalg as sparse_linalg
    except ImportError as exc:  # pragma: no cover - depends on optional scipy
        raise ImportError("Sparse mean-field inputs require scipy to be installed") from exc
    return sparse_linalg


def to_dense(matrix: Any) -> np.ndarray:
    if is_sparse_like(matrix):
        return np.asarray(matrix.toarray(), dtype=complex)
    return np.asarray(matrix, dtype=complex)


def as_sparse(matrix: Any):
    sparse = sparse_module()
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
        sparse = sparse_module()
        return sparse.bmat(
            [[as_sparse(top), None], [None, as_sparse(bottom)]],
            format="csr",
        )

    top = np.asarray(top, dtype=complex)
    bottom = np.asarray(bottom, dtype=complex)
    zero_top = np.zeros((top.shape[0], bottom.shape[1]), dtype=complex)
    zero_bottom = np.zeros((bottom.shape[0], top.shape[1]), dtype=complex)
    return np.block([[top, zero_top], [zero_bottom, bottom]])


def matrix_bound(matrix: Any) -> float:
    if is_sparse_like(matrix):
        row_sums = np.asarray(abs(matrix).sum(axis=1)).ravel()
        return float(np.max(row_sums)) if row_sums.size else 0.0
    array = np.asarray(matrix)
    if array.size == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(array), axis=1)))


def hermitian_spectral_bound(matrix: Any) -> float:
    if is_sparse_like(matrix):
        csr = as_sparse(matrix)
        nrows, ncols = csr.shape
        if nrows == 0 or ncols == 0:
            return 0.0
        if nrows != ncols:
            raise ValueError("matrix must be square")
        if nrows <= 64:
            eigenvalues = np.linalg.eigvalsh(to_dense(csr))
            return float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        try:
            sparse_linalg = sparse_linalg_module()
            largest = sparse_linalg.eigsh(
                csr,
                k=1,
                which="LA",
                return_eigenvectors=False,
                tol=1e-3,
                maxiter=max(10 * nrows, 1000),
            )
            smallest = sparse_linalg.eigsh(
                csr,
                k=1,
                which="SA",
                return_eigenvectors=False,
                tol=1e-3,
                maxiter=max(10 * nrows, 1000),
            )
            return float(
                max(
                    np.max(np.abs(np.asarray(largest, dtype=float))),
                    np.max(np.abs(np.asarray(smallest, dtype=float))),
                )
            )
        except Exception:
            return matrix_bound(csr)

    array = np.asarray(matrix, dtype=complex)
    if array.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(array)
    return float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
