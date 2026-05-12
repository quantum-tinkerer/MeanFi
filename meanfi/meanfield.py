from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import (
    _tb_type,
    add_tb,
    elementwise_product,
    is_sparse_like,
)
from meanfi.tb.validate import tb_dimension, zero_key


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the normal mean-field correction from a density matrix."""

    onsite_key = zero_key(tb_dimension(density_matrix))
    diagonal_density = np.real(
        np.diag(np.asarray(density_matrix[onsite_key], dtype=complex))
    )
    onsite_diagonal = np.zeros_like(diagonal_density, dtype=complex)
    sparse_present = any(is_sparse_like(matrix) for matrix in h_int.values())
    for vector in frozenset(h_int):
        interaction = h_int[vector]
        if is_sparse_like(interaction):
            onsite_diagonal += np.asarray(
                diagonal_density @ interaction, dtype=complex
            ).ravel()
        else:
            onsite_diagonal += diagonal_density @ np.asarray(interaction, dtype=complex)
    direct = {
        onsite_key: (
            sparse.diags(onsite_diagonal, format="csr")
            if sparse_present
            else np.diag(onsite_diagonal)
        )
    }
    exchange = {
        vector: -elementwise_product(h_int.get(vector, 0), density_matrix[vector])
        for vector in frozenset(h_int)
    }
    return add_tb(direct, exchange)


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


def assemble_bdg_correction(
    normal_block: _tb_type,
    anomalous_block: _tb_type,
    model,
) -> _tb_type:
    correction = assemble_bdg_tb(normal_block, anomalous_block, ndof=model._ndof)
    validate_bdg_tb(
        correction,
        ndof=model._ndof,
        ndim=model._ndim,
        name="BdG correction",
    )
    return correction


def bdg_correction_from_density_parts(
    density_matrix: _tb_type,
    *,
    h_int: _tb_type,
    ndof: int,
    ndim: int,
) -> _tb_type:
    electron_density = {
        key: matrix[:ndof, :ndof] for key, matrix in density_matrix.items()
    }
    anomalous_density = {
        key: matrix[:ndof, ndof:] for key, matrix in density_matrix.items()
    }
    normal_block = meanfield(electron_density, h_int)
    zero_e = np.zeros((ndof, ndof), dtype=complex)
    anomalous_block = {
        key: -elementwise_product(
            h_int.get(key, zero_e),
            anomalous_density.get(key, zero_e),
        )
        for key in frozenset(h_int) | frozenset(anomalous_density)
    }
    correction = assemble_bdg_tb(normal_block, anomalous_block, ndof=ndof)
    validate_bdg_tb(correction, ndof=ndof, ndim=ndim, name="BdG correction")
    return correction


def bdg_correction_from_density(density_matrix: _tb_type, model) -> _tb_type:
    return bdg_correction_from_density_parts(
        density_matrix,
        h_int=model.h_int,
        ndof=model._ndof,
        ndim=model._ndim,
    )


def bdg_density_keys(model, meanfield_correction: _tb_type) -> list[tuple[int, ...]]:
    del meanfield_correction
    keys = list(model.h_int)
    onsite = (0,) * model._ndim
    if onsite not in keys:
        keys.append(onsite)
    return keys
