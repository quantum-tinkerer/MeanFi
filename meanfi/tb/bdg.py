from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sparse

from meanfi.tb.ops import (
    _tb_type,
    as_sparse,
    block_diag,
    is_sparse_like,
    matrix_shape,
)
from meanfi.tb.validate import matrix_allclose
from meanfi.tb.storage import prefers_sparse_storage


def electron_to_bdg_tb(h: _tb_type, ndof: int) -> _tb_type:
    """Embed an electron-space tight-binding Hamiltonian into electron-first BdG space."""

    zero = (
        sparse.csr_matrix((ndof, ndof), dtype=complex)
        if prefers_sparse_storage(h)
        else np.zeros((ndof, ndof), dtype=complex)
    )
    keys = set(h)
    keys.update(tuple(-np.asarray(key, dtype=int)) for key in h)
    bdg = {}
    for key in keys:
        opposite = tuple(-np.asarray(key, dtype=int))
        top = h.get(key, zero)
        bottom = -h.get(opposite, zero).T
        bdg[key] = block_diag(top, bottom)
    return bdg


def split_bdg_matrix(matrix: Any, ndof: int) -> tuple[Any, Any, Any, Any]:
    return (
        matrix[:ndof, :ndof],
        matrix[:ndof, ndof:],
        matrix[ndof:, :ndof],
        matrix[ndof:, ndof:],
    )


def validate_bdg_tb(
    tb: _tb_type, *, ndof: int, ndim: int, name: str = "BdG correction"
) -> None:
    expected_shape = (2 * ndof, 2 * ndof)

    for key, matrix in tb.items():
        if len(key) != ndim:
            raise ValueError(f"{name} keys must match the model dimension")
        if matrix_shape(matrix) != expected_shape:
            raise ValueError(f"{name} matrices must have shape (2*ndof, 2*ndof)")

    for key, matrix in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        if opposite not in tb:
            raise ValueError(f"{name} must include opposite keys for Hermiticity")
        opposite_matrix = tb[opposite]
        if not matrix_allclose(matrix, opposite_matrix.conj().T):
            raise ValueError(
                f"{name} must be Hermitian in real-space tight-binding form"
            )

    for key, matrix in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        opposite_matrix = tb[opposite]
        _normal, anomalous, lower, hole = split_bdg_matrix(matrix, ndof)
        opposite_normal, opposite_anomalous, _, _ = split_bdg_matrix(
            opposite_matrix, ndof
        )

        if not matrix_allclose(hole, -opposite_normal.T):
            raise ValueError(
                f"{name} lower-right block must equal -h(-R).T in electron-first BdG form"
            )
        if not matrix_allclose(lower, opposite_anomalous.conj().T):
            raise ValueError(
                f"{name} lower-left block must equal Delta(-R).dagger in electron-first BdG form"
            )
        if not matrix_allclose(anomalous, -opposite_anomalous.T):
            raise ValueError(
                f"{name} anomalous block must satisfy Delta(R) = -Delta(-R).T"
            )


def particle_hole_conjugate(tb: _tb_type) -> _tb_type:
    result = {}
    for key, matrix in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        result[opposite] = -matrix.T
    return result


def assemble_bdg_tb(
    normal_block: _tb_type,
    anomalous_block: _tb_type,
    *,
    ndof: int,
) -> _tb_type:
    """Assemble electron and pairing blocks into electron-first BdG matrices."""

    zero = (
        sparse.csr_matrix((ndof, ndof), dtype=complex)
        if prefers_sparse_storage(normal_block, anomalous_block)
        else np.zeros((ndof, ndof), dtype=complex)
    )
    hole_block = particle_hole_conjugate(normal_block)
    keys = frozenset(normal_block) | frozenset(anomalous_block) | frozenset(hole_block)
    assembled = {}
    for key in keys:
        opposite = tuple(-np.asarray(key, dtype=int))
        normal = normal_block.get(key, zero)
        anomalous = anomalous_block.get(key, zero)
        lower = anomalous_block.get(opposite, zero).conj().T
        hole = hole_block.get(key, zero)
        if any(is_sparse_like(block) for block in (normal, anomalous, lower, hole)):
            assembled[key] = sparse.bmat(
                [
                    [as_sparse(normal), as_sparse(anomalous)],
                    [as_sparse(lower), as_sparse(hole)],
                ],
                format="csr",
            )
        else:
            assembled[key] = np.block(
                [
                    [
                        np.asarray(normal, dtype=complex),
                        np.asarray(anomalous, dtype=complex),
                    ],
                    [np.asarray(lower, dtype=complex), np.asarray(hole, dtype=complex)],
                ]
            )
    return assembled
