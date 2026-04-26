from __future__ import annotations

import numpy as np

from meanfi._finite_temp import fermi_dirac
from meanfi._info import DensityMatrixResult
from meanfi._validation import tb_dimension, zero_key
from meanfi.integration import IntegrationMethod, solve_density_matrix_at_mu
from meanfi.integration import solve_density_matrix_fixed_filling
from meanfi.tb.tb import add_tb, _tb_type

__all__ = [
    "DensityMatrixResult",
    "density_matrix",
    "density_matrix_at_mu",
    "fermi_dirac",
    "meanfield",
]


def _is_sparse_like(matrix) -> bool:
    return hasattr(matrix, "toarray") and hasattr(matrix, "tocsr")


def _sparse_module():
    try:
        import scipy.sparse as sparse
    except ImportError as exc:  # pragma: no cover - depends on optional scipy
        raise ImportError("Sparse mean-field inputs require scipy to be installed") from exc
    return sparse


def _elementwise_product(lhs, rhs):
    if _is_sparse_like(lhs):
        return lhs.multiply(np.asarray(rhs, dtype=complex)).tocsr()
    if _is_sparse_like(rhs):
        return rhs.multiply(np.asarray(lhs, dtype=complex)).tocsr()
    return np.asarray(lhs, dtype=complex) * np.asarray(rhs, dtype=complex)


def density_matrix_at_mu(
    h: _tb_type,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    *,
    integration: IntegrationMethod,
) -> DensityMatrixResult:
    """Compute the real-space density matrix at a fixed chemical potential."""

    return solve_density_matrix_at_mu(
        h,
        mu=mu,
        kT=kT,
        keys=keys,
        integration=integration,
    )


def density_matrix(
    h: _tb_type,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    *,
    integration: IntegrationMethod,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_mu_iterations: int | None = None,
) -> DensityMatrixResult:
    """Compute the fixed-filling real-space density matrix."""

    return solve_density_matrix_fixed_filling(
        h,
        filling=filling,
        kT=kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
    )


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the mean-field correction from a density matrix."""

    onsite_key = zero_key(tb_dimension(density_matrix))
    diagonal_density = np.real(np.diag(np.asarray(density_matrix[onsite_key], dtype=complex)))
    onsite_diagonal = np.zeros_like(diagonal_density, dtype=complex)
    sparse_present = any(_is_sparse_like(matrix) for matrix in h_int.values())
    sparse = _sparse_module() if sparse_present else None
    for vector in frozenset(h_int):
        interaction = h_int[vector]
        if _is_sparse_like(interaction):
            onsite_diagonal += np.asarray(diagonal_density @ interaction, dtype=complex).ravel()
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
        vector: -_elementwise_product(h_int.get(vector, 0), density_matrix[vector])
        for vector in frozenset(h_int)
    }
    return add_tb(direct, exchange)
