from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.integrate.occupations import fermi_dirac

from .base import ChebyshevFOE, _BlockResult
from .common import _derivative_convergence, gershgorin_bounds, matrix_action


def _chebyshev_coefficients(
    order: int,
    *,
    center: float,
    scale: float,
    kT: float,
    oversampling: int,
) -> np.ndarray:
    n_nodes = max(64, int(oversampling) * (order + 1))
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    values = fermi_dirac(scale * np.cos(theta) + center, kT, 0.0)
    coeffs = np.empty(order + 1, dtype=float)
    coeffs[0] = np.mean(values)
    for mode in range(1, order + 1):
        coeffs[mode] = 2.0 * np.mean(values * np.cos(mode * theta))
    return coeffs


def _apply_rescaled(matrix: Any, block: np.ndarray, *, center: float, scale: float) -> np.ndarray:
    return (matrix_action(matrix, block) - center * block) / scale


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
) -> _BlockResult:
    lower, upper = gershgorin_bounds(matrix)
    width = max(upper - lower, 1e-12)
    scale = 0.5 * width * (1.0 + options.spectral_padding)
    center = 0.5 * (upper + lower)

    accepted_block = None
    accepted_derivative = None
    accepted_error = float("inf")
    accepted_order = None
    derivative_converged = True

    order_half = int(options.initial_order)
    while 2 * order_half <= int(options.max_order):
        order = 2 * order_half
        coeffs = _chebyshev_coefficients(
            order,
            center=center,
            scale=scale,
            kT=kT,
            oversampling=options.coefficient_oversampling,
        )

        v_prev = np.array(block, copy=True)
        v_curr = _apply_rescaled(matrix, block, center=center, scale=scale)
        y_half = coeffs[0] * v_prev
        y_full = coeffs[0] * v_prev

        if derivative:
            s_prev = np.zeros_like(block)
            s_curr = -(q_diag[:, np.newaxis] * block) / scale
            dy_half = coeffs[0] * s_prev
            dy_full = coeffs[0] * s_prev
        else:
            s_prev = s_curr = None
            dy_half = dy_full = None

        if order_half >= 1:
            y_half = y_half + coeffs[1] * v_curr
        y_full = y_full + coeffs[1] * v_curr
        if derivative:
            if order_half >= 1:
                dy_half = dy_half + coeffs[1] * s_curr
            dy_full = dy_full + coeffs[1] * s_curr

        for mode in range(1, order):
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
            if term_index <= order_half:
                y_half = y_half + coeffs[term_index] * v_next
                if derivative:
                    dy_half = dy_half + coeffs[term_index] * s_next
            y_full = y_full + coeffs[term_index] * v_next
            if derivative:
                dy_full = dy_full + coeffs[term_index] * s_next

            v_prev, v_curr = v_curr, v_next
            if derivative:
                s_prev, s_curr = s_curr, s_next

        accepted_error = float(np.max(np.abs(y_full - y_half)))
        if derivative:
            derivative_converged, _derivative_error = _derivative_convergence(
                dy_full,
                dy_half,
                derivative=derivative,
                dn_dmu_rtol=options.dn_dmu_rtol,
                derivative_trace_monitor=derivative_trace_monitor,
                derivative_context=derivative_context,
                matrix_function_name="Chebyshev",
            )
        accepted_block = y_full
        accepted_derivative = dy_full
        accepted_order = order
        if accepted_error <= tolerance and derivative_converged:
            break
        order_half = order

    if accepted_error > tolerance or (derivative and not derivative_converged):
        raise ValueError("Chebyshev FOE did not converge within max_order")

    return _BlockResult(
        block=accepted_block,
        derivative_block=accepted_derivative,
        error=accepted_error,
        order=accepted_order,
    )
