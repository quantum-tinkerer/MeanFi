from __future__ import annotations

from typing import Any

import numpy as np

from .base import BdGMatrixFunction, ChebyshevFOE, DirectDiagonalization, RationalFOE, _BlockResult
from .chebyshev import _chebyshev_density_block
from .direct import _exact_density_block
from .rational import _rational_density_block


def resolve_matrix_function(selected: object | None) -> BdGMatrixFunction:
    if selected is None:
        return DirectDiagonalization()
    if not isinstance(selected, BdGMatrixFunction):
        raise TypeError("AdaptiveQuadrature.matrix_function must be a BdGMatrixFunction")
    return selected


def matrix_function_label(matrix_function: BdGMatrixFunction) -> str:
    if isinstance(matrix_function, ChebyshevFOE):
        return "Chebyshev FOE"
    if isinstance(matrix_function, RationalFOE):
        return "Rational FOE"
    if isinstance(matrix_function, DirectDiagonalization):
        return "direct diagonalization"
    return matrix_function.__class__.__name__


def density_block(
    matrix_function: BdGMatrixFunction,
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    derivative_trace_monitor=None,
    derivative_context: str | None = None,
) -> _BlockResult:
    if isinstance(matrix_function, DirectDiagonalization):
        return _exact_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
        )
    if isinstance(matrix_function, ChebyshevFOE):
        return _chebyshev_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
            tolerance=tolerance,
            options=matrix_function,
            derivative_trace_monitor=derivative_trace_monitor,
            derivative_context=derivative_context,
        )
    if isinstance(matrix_function, RationalFOE):
        return _rational_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
            tolerance=tolerance,
            options=matrix_function,
            derivative_trace_monitor=derivative_trace_monitor,
            derivative_context=derivative_context,
        )
    raise TypeError("matrix_function must be a BdGMatrixFunction instance")
