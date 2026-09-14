from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse

from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import (
    _tb_type,
    add_tb,
    as_sparse,
    elementwise_product,
    is_sparse_like,
    transpose,
)
from meanfi.tb.validate import tb_dimension, zero_key
from meanfi.tb.storage import prefers_sparse_storage

if TYPE_CHECKING:
    from meanfi.model import Model


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the normal mean-field correction from a density matrix."""

    onsite_key = zero_key(tb_dimension(density_matrix))
    diagonal_density = np.asarray(density_matrix[onsite_key].diagonal()).real.ravel()
    onsite_diagonal = np.zeros_like(diagonal_density, dtype=complex)
    sparse_present = prefers_sparse_storage(density_matrix, h_int)
    for interaction in h_int.values():
        onsite_diagonal += np.asarray(diagonal_density @ interaction).ravel()
    direct = {
        onsite_key: (
            sparse.diags(onsite_diagonal, format="csr")
            if sparse_present
            else np.diag(onsite_diagonal)
        )
    }
    exchange = {
        key: -elementwise_product(interaction, density_matrix[key])
        for key, interaction in h_int.items()
    }
    return add_tb(direct, exchange)


def extract_electron_density(density_matrix: _tb_type, model: Model) -> _tb_type:
    return {
        key: matrix[: model._ndof, : model._ndof]
        for key, matrix in density_matrix.items()
    }


def extract_anomalous_density(density_matrix: _tb_type, model: Model) -> _tb_type:
    return {
        key: matrix[: model._ndof, model._ndof :]
        for key, matrix in density_matrix.items()
    }


def _antisymmetrize_anomalous_block(anomalous_block: _tb_type, ndof: int) -> _tb_type:
    """Project pairing blocks onto Delta(R) = -Delta(-R).T."""

    zero = (
        sparse.csr_matrix((ndof, ndof), dtype=complex)
        if prefers_sparse_storage(anomalous_block)
        else np.zeros((ndof, ndof), dtype=complex)
    )
    keys = frozenset(anomalous_block) | {
        tuple(-np.asarray(key, dtype=int)) for key in anomalous_block
    }
    result = {}
    for key in keys:
        if key in result:
            continue
        opposite = tuple(-np.asarray(key, dtype=int))
        block = anomalous_block.get(key, zero)
        opposite_block = anomalous_block.get(opposite, zero)
        if is_sparse_like(block) or is_sparse_like(opposite_block):
            projected = 0.5 * (as_sparse(block) - transpose(as_sparse(opposite_block)))
            projected = projected.tocsr()
        else:
            projected = 0.5 * (
                np.asarray(block, dtype=complex)
                - transpose(np.asarray(opposite_block, dtype=complex))
            )
        result[key] = projected
        if opposite != key:
            result[opposite] = -transpose(projected)
    return result


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
    anomalous_block = {
        key: -elementwise_product(
            interaction,
            anomalous_density[key],
        )
        for key, interaction in h_int.items()
    }
    anomalous_block = _antisymmetrize_anomalous_block(anomalous_block, ndof)
    correction = assemble_bdg_tb(normal_block, anomalous_block, ndof=ndof)
    validate_bdg_tb(correction, ndof=ndof, ndim=ndim, name="BdG correction")
    return correction


def bdg_correction_from_density(density_matrix: _tb_type, model: Model) -> _tb_type:
    return bdg_correction_from_density_parts(
        density_matrix,
        h_int=model.h_int,
        ndof=model._ndof,
        ndim=model._ndim,
    )
