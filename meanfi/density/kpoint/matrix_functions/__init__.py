"""Matrix-function settings and their resolution at the density boundary."""

from .base import DirectDiagonalization, RationalFOE
from meanfi.tb.ops import _tb_type, is_sparse_like


def resolve_periodic_matrix_function(
    selected: object | None,
    hamiltonian: _tb_type,
    *,
    kT: float,
    prescribed: bool,
) -> DirectDiagonalization | RationalFOE:
    sparse = any(is_sparse_like(matrix) for matrix in hamiltonian.values())
    if selected is None:
        if sparse:
            if prescribed and kT > 0:
                return RationalFOE()
            raise ValueError(
                "Automatic sparse UniformGrid evaluation requires kT > 0 and "
                "prescribed nk. Use UniformGrid(nk=..., matrix_function=RationalFOE()) "
                "or explicitly choose DirectDiagonalization() to permit dense batches."
            )
        return DirectDiagonalization()
    resolved = selected
    if isinstance(resolved, RationalFOE):
        if not prescribed:
            raise ValueError(
                "UniformGrid RationalFOE requires prescribed nk; adaptive RationalFOE is unsupported"
            )
        if kT <= 0:
            raise ValueError("UniformGrid RationalFOE requires kT > 0")
        if not sparse:
            raise ValueError(
                "UniformGrid RationalFOE is supported only for sparse matrices"
            )
    elif not isinstance(resolved, DirectDiagonalization):
        raise TypeError(
            "UniformGrid.matrix_function must be DirectDiagonalization or RationalFOE"
        )
    return resolved


__all__ = ["DirectDiagonalization", "RationalFOE"]
