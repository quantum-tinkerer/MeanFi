from __future__ import annotations

import math

from meanfi.tb.ops import is_sparse_like, _tb_type

from meanfi.density.kpoint.matrix_functions import DirectDiagonalization
from .methods import PeriodicGrid, AdaptiveSimplex, IntegrationMethod


DEFAULT_KT = 0.0


def uses_sparse_matrices(hamiltonian: _tb_type) -> bool:
    return any(is_sparse_like(matrix) for matrix in hamiltonian.values())


def select_default_integration(
    hamiltonian: _tb_type,
    *,
    kT: float,
    superconducting: bool = False,
) -> IntegrationMethod:
    if not math.isfinite(kT) or kT < 0:
        raise ValueError(
            "meanfi supports only finite non-negative temperatures (kT >= 0)"
        )

    if kT == 0:
        if superconducting:
            raise NotImplementedError(
                "Zero-temperature superconducting calculations require an explicit "
                "PeriodicGrid(nk=...) integration setting."
            )
        return AdaptiveSimplex()

    if uses_sparse_matrices(hamiltonian):
        raise ValueError(
            "Automatic finite-temperature sparse integration is no longer supported. "
            "Use PeriodicGrid(nk=..., matrix_function=RationalFOE()) for a prescribed "
            "sparse mesh, or explicitly select PeriodicGrid(matrix_function="
            "DirectDiagonalization()) if dense diagonalization fits in memory."
        )

    return PeriodicGrid(
        matrix_function=DirectDiagonalization(),
    )


__all__ = ["DEFAULT_KT", "select_default_integration", "uses_sparse_matrices"]
