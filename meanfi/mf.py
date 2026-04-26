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
    direct = {
        onsite_key: np.sum(
            np.asarray(
                [
                    np.diag(
                        np.einsum("pp,pn->n", density_matrix[onsite_key], h_int[vector])
                    )
                    for vector in frozenset(h_int)
                ]
            ),
            axis=0,
        )
    }
    exchange = {
        vector: -1 * h_int.get(vector, 0) * density_matrix[vector]
        for vector in frozenset(h_int)
    }
    return add_tb(direct, exchange)
