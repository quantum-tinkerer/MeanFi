from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import anderson

from meanfi.scf.methods import AndersonMixing, LinearMixing, SCFMethod


class NoConvergence(Exception):
    """Raised when the self-consistent field solver does not converge."""

    def __init__(self, last_iterate: np.ndarray):
        self.last_iterate = np.array(last_iterate, copy=True)
        super().__init__(self.last_iterate)


def max_norm(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def translate_no_convergence(exc: Exception, fallback: np.ndarray) -> None:
    if exc.__class__.__name__ != "NoConvergence":
        return

    if hasattr(exc, "last_iterate"):
        last_iterate = exc.last_iterate
    elif exc.args:
        last_iterate = exc.args[0]
    else:
        last_iterate = fallback
    raise NoConvergence(np.asarray(last_iterate, dtype=float)) from exc


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
    for iteration in range(1, maxiter + 1):
        residual = np.asarray(residual_fn(x), dtype=float)
        residual_norm = max_norm(residual)
        on_iteration(iteration, residual_norm)
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
    state = {"iterations": 0}

    def optimizer_callback(x: np.ndarray, f: np.ndarray) -> None:
        del x
        state["iterations"] += 1
        on_iteration(state["iterations"], max_norm(np.asarray(f, dtype=float)))

    try:
        with np.errstate(invalid="ignore"):
            result = anderson(
                residual_fn,
                x0,
                callback=optimizer_callback,
                M=int(scf.M),
                line_search=scf.line_search,
                maxiter=int(scf.max_iterations),
                f_tol=scf_tol,
                tol_norm=max_norm,
            )
    except Exception as exc:  # pragma: no cover - exercised through scipy
        translate_no_convergence(exc, x0)
        raise

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
