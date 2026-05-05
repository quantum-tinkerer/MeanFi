from __future__ import annotations

import numpy as np

from meanfi.tb.ops import elementwise_product, is_sparse_like, sparse_module
from meanfi.tb.validate import tb_dimension, zero_key
from meanfi.tb.ops import add_tb, _tb_type


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the mean-field correction from a density matrix."""

    onsite_key = zero_key(tb_dimension(density_matrix))
    diagonal_density = np.real(
        np.diag(np.asarray(density_matrix[onsite_key], dtype=complex))
    )
    onsite_diagonal = np.zeros_like(diagonal_density, dtype=complex)
    sparse_present = any(is_sparse_like(matrix) for matrix in h_int.values())
    sparse = sparse_module() if sparse_present else None
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
