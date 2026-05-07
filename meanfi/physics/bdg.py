from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sparse

from meanfi.integrate.filling import _conservative_spectral_bound
from meanfi.tb.ops import (
    as_sparse,
    block_diag,
    conjugate_transpose,
    elementwise_product,
    is_sparse_like,
    matrix_shape,
    transpose,
)
from meanfi.tb.validate import matrix_allclose
from meanfi.physics.meanfield import meanfield as normal_meanfield
from meanfi.tb.ops import _tb_type


def electron_to_bdg_tb(h: _tb_type, ndof: int) -> _tb_type:
    """Embed an electron-space tight-binding Hamiltonian into electron-first BdG space."""

    zero = np.zeros((ndof, ndof), dtype=complex)
    keys = set(h)
    keys.update(tuple(-np.asarray(key, dtype=int)) for key in h)
    bdg = {}
    for key in keys:
        opposite = tuple(-np.asarray(key, dtype=int))
        top = h.get(key, zero)
        bottom = -transpose(h.get(opposite, zero))
        bdg[key] = block_diag(top, bottom)
    return bdg


def charge_diagonal(ndof: int) -> np.ndarray:
    """Return the electron-first Nambu charge diagonal."""

    return np.concatenate([np.ones(ndof), -np.ones(ndof)])


def mu_bracket_for_bdg(hamiltonian: _tb_type, kT: float) -> tuple[float, float]:
    bound = _conservative_spectral_bound(hamiltonian)
    padding = max(1.0, 10.0 * kT)
    return -float(bound + padding), float(bound + padding)


def split_bdg_matrix(matrix: Any, ndof: int) -> tuple[Any, Any, Any, Any]:
    array = matrix
    return (
        array[:ndof, :ndof],
        array[:ndof, ndof:],
        array[ndof:, :ndof],
        array[ndof:, ndof:],
    )


def validate_bdg_tb(
    tb: _tb_type, *, ndof: int, ndim: int, name: str = "BdG correction"
) -> None:
    expected_shape = (2 * ndof, 2 * ndof)
    zero = np.zeros(expected_shape, dtype=complex)

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
        if not matrix_allclose(matrix, conjugate_transpose(opposite_matrix)):
            raise ValueError(
                f"{name} must be Hermitian in real-space tight-binding form"
            )

    keys = frozenset(tb) | {tuple(-np.asarray(key, dtype=int)) for key in tb}
    for key in keys:
        opposite = tuple(-np.asarray(key, dtype=int))
        matrix = tb.get(key, zero)
        opposite_matrix = tb.get(opposite, zero)
        normal, anomalous, lower, hole = split_bdg_matrix(matrix, ndof)
        opposite_normal, opposite_anomalous, _, _ = split_bdg_matrix(
            opposite_matrix, ndof
        )

        if not matrix_allclose(hole, -transpose(opposite_normal)):
            raise ValueError(
                f"{name} lower-right block must equal -h(-R).T in electron-first BdG form"
            )
        if not matrix_allclose(lower, conjugate_transpose(opposite_anomalous)):
            raise ValueError(
                f"{name} lower-left block must equal Delta(-R).dagger in electron-first BdG form"
            )


def zero_bdg_array(ndof: int) -> np.ndarray:
    return np.zeros((2 * ndof, 2 * ndof), dtype=complex)


def zero_electron_matrix(model) -> np.ndarray:
    return np.zeros((model._ndof, model._ndof), dtype=complex)


def extract_electron_density(density_matrix: _tb_type, model) -> _tb_type:
    return {
        key: matrix[: model._ndof, : model._ndof]
        for key, matrix in density_matrix.items()
    }


def extract_anomalous_density(density_matrix: _tb_type, model) -> _tb_type:
    return {
        key: matrix[: model._ndof, model._ndof :]
        for key, matrix in density_matrix.items()
    }


def particle_hole_conjugate(tb: _tb_type) -> _tb_type:
    result = {}
    for key, matrix in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        result[opposite] = -transpose(matrix)
    return result


def assemble_bdg_correction(
    normal_block: _tb_type,
    anomalous_block: _tb_type,
    model,
) -> _tb_type:
    zero_e = zero_electron_matrix(model)
    keys = frozenset(normal_block) | frozenset(anomalous_block)
    hole_block = particle_hole_conjugate(normal_block)
    assembled = {}
    for key in keys | frozenset(hole_block):
        opposite = tuple(-np.asarray(key, dtype=int))
        normal = normal_block.get(key, zero_e)
        anomalous = anomalous_block.get(key, zero_e)
        lower = conjugate_transpose(anomalous_block.get(opposite, zero_e))
        hole = hole_block.get(key, zero_e)
        if is_sparse_like(normal) or is_sparse_like(anomalous) or is_sparse_like(hole):
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


def bdg_correction_from_density(density_matrix: _tb_type, model) -> _tb_type:
    electron_density = extract_electron_density(density_matrix, model)
    anomalous_density = extract_anomalous_density(density_matrix, model)
    normal_block = normal_meanfield(electron_density, model.h_int)
    zero_e = zero_electron_matrix(model)
    anomalous_block = {
        key: -elementwise_product(
            model.h_int.get(key, zero_e),
            anomalous_density.get(key, zero_e),
        )
        for key in frozenset(model.h_int) | frozenset(anomalous_density)
    }
    return assemble_bdg_correction(normal_block, anomalous_block, model)


def bdg_density_keys(model, meanfield: _tb_type) -> list[tuple[int, ...]]:
    del meanfield
    keys = list(model.h_int)
    onsite = (0,) * model._ndim
    if onsite not in keys:
        keys.append(onsite)
    return keys
