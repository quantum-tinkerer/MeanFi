from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import NoConvergence as ScipyNoConvergence, anderson

from meanfi.errors import ConvergenceError
from meanfi.scf.methods import AndersonMixing, LinearMixing, SCFMethod

if TYPE_CHECKING:
    from meanfi.results import SCFResult


class SolverError(ConvergenceError):
    """Base class for solver failures with an optional last valid result."""

    def __init__(self, message: str, *, result: SCFResult | None = None):
        self.result = result
        super().__init__(message)


class NoConvergence(SolverError):
    """Raised when the SCF iteration budget is exhausted."""

    def __init__(
        self,
        last_iterate: np.ndarray,
        *,
        result: SCFResult | None = None,
    ):
        self.last_iterate = np.array(last_iterate, dtype=float, copy=True)
        super().__init__(
            "self-consistent field iteration did not converge", result=result
        )


class SolverFailure(SolverError):
    """Raised when a numerical evaluation fails; result is None before the first valid state."""


def max_norm(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _solve_linear_mixing(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    alpha: float,
    maxiter: int,
    scf_tol: float,
    on_iteration,
) -> np.ndarray:
    x = np.array(x0, copy=True)
    for _ in range(maxiter):
        residual = np.asarray(residual_fn(x), dtype=float)
        residual_norm = max_norm(residual)
        on_iteration(x, residual)
        if residual_norm <= scf_tol:
            return x
        x = x + alpha * residual
    raise NoConvergence(x)


def _solve_anderson(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    scf: AndersonMixing,
    scf_tol: float,
    on_iteration,
) -> np.ndarray:
    accepted_initial = False
    accepted = 0

    def accept(x, residual):
        nonlocal accepted
        accepted += 1
        on_iteration(x, residual)
        if accepted >= scf.max_iterations and max_norm(residual) > scf_tol:
            raise NoConvergence(x)

    def wrapped_residual_fn(x: np.ndarray) -> np.ndarray:
        nonlocal accepted_initial
        residual = np.asarray(residual_fn(x), dtype=float)
        if not accepted_initial:
            accepted_initial = True
            accept(x, residual)
        return residual

    try:
        with np.errstate(invalid="ignore"):
            result = anderson(
                wrapped_residual_fn,
                x0,
                callback=accept,
                alpha=scf.alpha,
                w0=scf.regularization,
                M=scf.history_size,
                line_search=scf.line_search,
                maxiter=int(scf.max_iterations),
                f_tol=scf_tol,
                tol_norm=max_norm,
            )
    except ScipyNoConvergence as exc:
        raise NoConvergence(exc.args[0]) from exc

    return np.asarray(result, dtype=float)


def solve_fixed_point(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    scf: SCFMethod,
    scf_tol: float,
    on_iteration,
) -> np.ndarray:
    if isinstance(scf, LinearMixing):
        return _solve_linear_mixing(
            residual_fn,
            x0,
            alpha=float(scf.alpha),
            maxiter=int(scf.max_iterations),
            scf_tol=scf_tol,
            on_iteration=on_iteration,
        )
    if isinstance(scf, AndersonMixing):
        return _solve_anderson(
            residual_fn,
            x0,
            scf=scf,
            scf_tol=scf_tol,
            on_iteration=on_iteration,
        )
    raise TypeError("scf must be an SCFMethod instance")
