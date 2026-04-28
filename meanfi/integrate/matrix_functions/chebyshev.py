from __future__ import annotations

from typing import Any

import numpy as np

from .base import ChebyshevFOE, _BlockResult
from .common import (
    _derivative_convergence,
    chebyshev_foe_coefficients,
    matrix_action,
    spectral_interval,
    workspace_matrix,
)


def _apply_rescaled(matrix: Any, block: np.ndarray, *, center: float, scale: float) -> np.ndarray:
    return (matrix_action(matrix, block) - center * block) / scale


def _chebyshev_density_block_at_order(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    order: int,
    options: ChebyshevFOE,
    derivative: bool,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> tuple[np.ndarray, np.ndarray | None]:
    matrix = workspace_matrix(matrix, workspace_dtype)
    block = np.asarray(block, dtype=workspace_dtype)
    lower, upper = spectral_interval(
        matrix,
        spectral_padding=options.spectral_padding,
    )
    width = max(upper - lower, 1e-12)
    scale = 0.5 * width
    center = 0.5 * (upper + lower)
    coeffs = chebyshev_foe_coefficients(
        int(order),
        center=center,
        scale=scale,
        kT=kT,
        mu=0.0,
        oversampling=options.coefficient_oversampling,
        derivative=False,
    )

    v_prev = np.array(block, copy=True)
    y = coeffs[0] * v_prev
    if derivative:
        s_prev = np.zeros_like(block)
        dy = coeffs[0] * s_prev
    else:
        s_prev = None
        dy = None

    if int(order) == 0:
        return y, dy

    v_curr = _apply_rescaled(matrix, block, center=center, scale=scale)
    y = y + coeffs[1] * v_curr
    if derivative:
        s_curr = -(q_diag[:, np.newaxis] * block) / scale
        dy = dy + coeffs[1] * s_curr
    else:
        s_curr = None

    for mode in range(1, int(order)):
        v_next = 2.0 * _apply_rescaled(
            matrix,
            v_curr,
            center=center,
            scale=scale,
        ) - v_prev
        if derivative:
            d_x_v = -(q_diag[:, np.newaxis] * v_curr) / scale
            s_next = (
                2.0
                * _apply_rescaled(
                    matrix,
                    s_curr,
                    center=center,
                    scale=scale,
                )
                + 2.0 * d_x_v
                - s_prev
            )

        term_index = mode + 1
        y = y + coeffs[term_index] * v_next
        if derivative:
            dy = dy + coeffs[term_index] * s_next

        v_prev, v_curr = v_curr, v_next
        if derivative:
            s_prev, s_curr = s_curr, s_next

    return y, dy


def _chebyshev_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    options: ChebyshevFOE,
    derivative_trace_monitor=None,
    derivative_context: str | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> _BlockResult:
    accepted_block = None
    accepted_derivative = None
    accepted_error = float("inf")
    accepted_order = None
    derivative_converged = True

    order_half = int(options.initial_order)
    matrix = workspace_matrix(matrix, workspace_dtype)
    block = np.asarray(block, dtype=workspace_dtype)
    half_block, half_derivative = _chebyshev_density_block_at_order(
        matrix,
        block,
        kT=kT,
        q_diag=q_diag,
        order=order_half,
        options=options,
        derivative=derivative,
        workspace_dtype=workspace_dtype,
    )
    while 2 * order_half <= int(options.max_order):
        order = 2 * order_half
        full_block, full_derivative = _chebyshev_density_block_at_order(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            order=order,
            options=options,
            derivative=derivative,
            workspace_dtype=workspace_dtype,
        )
        accepted_error = float(np.max(np.abs(full_block - half_block)))
        if derivative:
            derivative_converged, _derivative_error = _derivative_convergence(
                full_derivative,
                half_derivative,
                derivative=derivative,
                dn_dmu_rtol=options.dn_dmu_rtol,
                derivative_trace_monitor=derivative_trace_monitor,
                derivative_context=derivative_context,
                matrix_function_name="Chebyshev",
            )
        accepted_block = full_block
        accepted_derivative = full_derivative
        accepted_order = order
        if accepted_error <= tolerance and derivative_converged:
            break
        order_half = order
        half_block = full_block
        half_derivative = full_derivative

    if accepted_error > tolerance or (derivative and not derivative_converged):
        raise ValueError("Chebyshev FOE did not converge within max_order")

    return _BlockResult(
        block=accepted_block,
        derivative_block=accepted_derivative,
        error=accepted_error,
        order=accepted_order,
    )
