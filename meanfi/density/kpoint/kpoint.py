"""Single-k density evaluation entrypoint.

The BZ integration layer calls into this level whenever it needs the matrix
density contribution at one sampled Hamiltonian. Direct diagonalization,
rational FOE, and sparse selected-inverse details live one level lower.
"""

from __future__ import annotations

from meanfi.density.kpoint.matrix_functions import (
    BdGMatrixFunction,
    DirectDiagonalization,
    RationalFOE,
    density_block,
)
from meanfi.density.kpoint.occupations import fermi_dirac
from meanfi.density.kpoint.zero_dim import (
    density_matrix_at_mu_zero_dim,
    density_matrix_zero_dim,
    zero_dim_zero_temp_mu,
)

__all__ = [
    "BdGMatrixFunction",
    "DirectDiagonalization",
    "RationalFOE",
    "density_block",
    "density_matrix_at_mu_zero_dim",
    "density_matrix_zero_dim",
    "fermi_dirac",
    "zero_dim_zero_temp_mu",
]
