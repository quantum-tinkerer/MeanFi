from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mumps
import numpy as np
import scipy.sparse as sparse

from meanfi.tb.ops import as_sparse


@dataclass(frozen=True)
class SelectedInversePattern:
    size: int
    indptr: np.ndarray
    indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    fortran_indptr: np.ndarray
    fortran_indices: np.ndarray
    lookup: dict[tuple[int, int], int]

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def value_map(self, values: np.ndarray) -> dict[tuple[int, int], complex]:
        return {
            (int(row), int(col)): complex(value)
            for row, col, value in zip(self.rows, self.cols, values, strict=True)
        }


def build_selected_inverse_pattern(
    *,
    size: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> SelectedInversePattern:
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.size != cols.size:
        raise ValueError("rows and cols must have the same size")

    if rows.size == 0:
        pattern_matrix = sparse.csc_matrix((size, size), dtype=np.complex128)
    else:
        pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
        pattern_matrix = sparse.csc_matrix(
            (
                np.ones(pairs.shape[0], dtype=np.complex128),
                (pairs[:, 0], pairs[:, 1]),
            ),
            shape=(size, size),
        )
    pattern_matrix.sort_indices()

    pattern_rows = np.asarray(pattern_matrix.indices, dtype=int)
    pattern_cols = np.repeat(
        np.arange(size, dtype=int),
        np.diff(np.asarray(pattern_matrix.indptr, dtype=int)),
    )
    lookup = {
        (int(row), int(col)): position
        for position, (row, col) in enumerate(
            zip(pattern_rows, pattern_cols, strict=True)
        )
    }
    indptr = np.asarray(pattern_matrix.indptr, dtype=np.int32)
    indices = np.asarray(pattern_matrix.indices, dtype=np.int32)
    return SelectedInversePattern(
        size=int(size),
        indptr=indptr,
        indices=indices,
        rows=pattern_rows,
        cols=pattern_cols,
        fortran_indptr=np.asfortranarray(indptr) + 1,
        fortran_indices=np.asfortranarray(indices) + 1,
        lookup=lookup,
    )


class SelectedInverseFactorization:
    def __init__(self) -> None:
        self._context = mumps.Context(verbose=False)
        self._analysis_ready = False

    def factor(self, matrix: Any) -> None:
        shifted = as_sparse(matrix).tocsc().astype(np.complex128, copy=False)
        if self._analysis_ready:
            self._context.factor(shifted, reuse_analysis=True)
        else:
            self._context.factor(shifted)
            self._analysis_ready = True

    def selected_inverse(self, pattern: SelectedInversePattern) -> np.ndarray:
        if pattern.nnz == 0:
            return np.empty(0, dtype=np.complex128)
        values = np.zeros(pattern.nnz, dtype=np.complex128, order="F")
        instance = self._context.mumps_instance
        instance.set_sparse_rhs(
            pattern.fortran_indptr,
            pattern.fortran_indices,
            values,
        )
        instance.icntl[20] = 1
        instance.icntl[30] = 1
        instance.job = 3
        self._context.call()
        return np.asarray(values, dtype=np.complex128)
