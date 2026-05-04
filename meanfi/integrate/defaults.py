from __future__ import annotations

from meanfi.tb.ops import is_sparse_like, _tb_type

from .matrix_functions import DirectDiagonalization, RationalFOE
from .methods import AdaptiveQuadrature, AdaptiveSimplex, IntegrationMethod


DEFAULT_KT = 0.0


def uses_sparse_matrices(hamiltonian: _tb_type) -> bool:
    return any(is_sparse_like(matrix) for matrix in hamiltonian.values())


def select_default_integration(
    hamiltonian: _tb_type,
    *,
    kT: float,
    superconducting: bool = False,
) -> IntegrationMethod:
    if kT < 0:
        raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")

    if kT == 0:
        if superconducting:
            raise NotImplementedError(
                "Zero-temperature superconducting calculations require an explicit "
                "UniformGrid(...) integration setting."
            )
        return AdaptiveSimplex()

    if uses_sparse_matrices(hamiltonian):
        return AdaptiveQuadrature(
            matrix_function=RationalFOE(rational_scheme="aaa"),
        )

    return AdaptiveQuadrature(
        matrix_function=DirectDiagonalization(),
    )


__all__ = ["DEFAULT_KT", "select_default_integration", "uses_sparse_matrices"]
