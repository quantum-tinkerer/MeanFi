from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.tb.ops import is_sparse_like

from .base import BdGMatrixFunction, DirectDiagonalization, RationalFOE, _BlockResult
from .direct import _exact_density_block
from .rational.common import _rational_density_block


def resolve_matrix_function(selected: object | None) -> BdGMatrixFunction:
    if selected is None:
        return DirectDiagonalization()
    if not isinstance(selected, BdGMatrixFunction):
        raise TypeError(
            "AdaptiveQuadrature.matrix_function must be a BdGMatrixFunction"
        )
    return selected


def matrix_function_label(matrix_function: BdGMatrixFunction) -> str:
    if isinstance(matrix_function, RationalFOE):
        return "Rational FOE"
    if isinstance(matrix_function, DirectDiagonalization):
        return "direct diagonalization"
    return matrix_function.__class__.__name__


def resolve_sparse_default_matrix_function(
    selected: object | None,
    hamiltonian,
    *,
    parameter_name: str,
) -> DirectDiagonalization | RationalFOE:
    if selected is None and any(
        is_sparse_like(matrix) for matrix in hamiltonian.values()
    ):
        return RationalFOE(rational_scheme="aaa")
    resolved = resolve_matrix_function(selected)
    if not isinstance(resolved, (DirectDiagonalization, RationalFOE)):
        raise TypeError(
            f"{parameter_name} must be DirectDiagonalization or RationalFOE"
        )
    return resolved


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
    workspace_dtype: np.dtype = np.dtype(complex),
) -> _BlockResult:
    if isinstance(matrix_function, DirectDiagonalization):
        return _exact_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
            workspace_dtype=workspace_dtype,
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
            workspace_dtype=workspace_dtype,
        )
    raise TypeError("matrix_function must be a BdGMatrixFunction instance")
