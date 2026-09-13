from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sparse

from meanfi.tb.ops import is_sparse_like


def shift_by_mu(
    matrix: Any, mu: float, q_diag: np.ndarray, *, dtype: np.dtype | None = None
):
    resolved_dtype = np.dtype(complex) if dtype is None else np.dtype(dtype)
    if is_sparse_like(matrix):
        shifted = matrix.astype(resolved_dtype, copy=False)
        return shifted - float(mu) * sparse.diags(
            q_diag.astype(resolved_dtype), format="csr"
        )
    return np.asarray(matrix, dtype=resolved_dtype) - float(mu) * np.diag(
        q_diag.astype(resolved_dtype)
    )


def spectral_interval(matrix: Any) -> tuple[float, float]:
    """Gershgorin enclosure without eigensolves or sparse-to-dense conversion."""
    if is_sparse_like(matrix):
        diagonal = np.asarray(matrix.diagonal(), dtype=complex)
        row_sums = np.asarray(abs(matrix).sum(axis=1)).ravel()
    else:
        array = np.asarray(matrix, dtype=complex)
        diagonal = np.diag(array)
        row_sums = np.sum(np.abs(array), axis=1)

    radius = np.maximum(row_sums - np.abs(diagonal), 0.0)
    center = diagonal.real
    return float(np.min(center - radius)), float(np.max(center + radius))
