from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import NoConvergence as ScipyNoConvergence, anderson

from meanfi.errors import ConvergenceError
from meanfi.scf.methods import AndersonMixing

if TYPE_CHECKING:
    from meanfi.results import SCFResult
    from meanfi.scf.problem import SCFEvaluation


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


def iterate_anderson(
    evaluate: Callable[[np.ndarray], SCFEvaluation],
    x0: np.ndarray,
    *,
    scf: AndersonMixing,
    scf_tol: float,
    accept: Callable[[SCFEvaluation], None],
) -> None:
    """Adapt SciPy's residual/callback interface to complete SCF evaluations."""
    trials: list[SCFEvaluation] = []
    accepted = 0
    last = None

    def commit(trial):
        nonlocal accepted, last
        accept(trial)
        last = trial
        trials.clear()
        accepted += 1
        if accepted >= scf.max_iterations and trial.residual_norm > scf_tol:
            raise NoConvergence(trial.input_state.values)

    def residual(x):
        trial = evaluate(x)
        trials.append(trial)
        if accepted == 0:
            # SciPy omits its callback for the initial evaluation.
            commit(trial)
        return trial.residual

    def on_iteration(x, residual):
        trial = next(
            (
                trial
                for trial in reversed(trials)
                if np.array_equal(trial.input_state.values, np.ravel(x))
                and np.array_equal(trial.residual, np.ravel(residual))
            ),
            None,
        )
        commit(evaluate(x) if trial is None else trial)

    try:
        with np.errstate(invalid="ignore"):
            result = anderson(
                residual,
                x0,
                callback=on_iteration,
                alpha=scf.alpha,
                w0=scf.regularization,
                M=scf.history_size,
                line_search=scf.line_search,
                maxiter=scf.max_iterations,
                f_tol=scf_tol,
                tol_norm=max_norm,
            )
    except ScipyNoConvergence as exc:
        raise NoConvergence(exc.args[0]) from exc
    if last is None or not np.array_equal(result, last.input_state.values):
        commit(evaluate(result))
