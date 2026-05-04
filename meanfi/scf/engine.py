from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from meanfi.results import DensityMatrixResult, SCFInfo, SolverResult
from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like, to_dense

from .methods import AndersonMixing, LinearMixing, SCFMethod


class NoConvergence(Exception):
    """Raised when the self-consistent field solver does not converge."""

    def __init__(self, last_iterate: np.ndarray):
        self.last_iterate = np.array(last_iterate, copy=True)
        super().__init__(self.last_iterate)


@dataclass(frozen=True)
class SolverRuntime:
    integration: object
    filling_tol: float | None
    mu_tol: float
    max_mu_iterations: int | None


@dataclass
class SCFRunState:
    iterations: int = 0
    mu: float = 0.0
    density_matrix_result: DensityMatrixResult | None = None
    residual_norm: float = float("inf")
    total_charge_integration_calls: int = 0
    total_density_integration_calls: int = 0
    total_kernel_evals: int = 0
    total_unique_evals: int = 0
    total_evaluator_evals: int = 0


@dataclass
class SCFRunResult:
    params: np.ndarray
    final_density_result: DensityMatrixResult
    state: SCFRunState
    residual_norm: float


def _prefer_sparse(*tb_dicts: _tb_type) -> bool:
    return any(
        is_sparse_like(matrix)
        for tb in tb_dicts
        if tb is not None
        for matrix in tb.values()
    )


def restore_tb_type(tb: _tb_type, *, prefer_sparse: bool) -> _tb_type:
    if not prefer_sparse:
        return tb
    return {key: as_sparse(value) for key, value in tb.items()}


def warn_on_projection(original: _tb_type, projected: _tb_type, *, label: str) -> None:
    import warnings

    for key in frozenset(original) | frozenset(projected):
        before = to_dense(original.get(key, np.zeros((0, 0), dtype=complex)))
        after = to_dense(projected.get(key, np.zeros((0, 0), dtype=complex)))
        if before.shape != after.shape:
            continue
        if np.any(np.abs(before - after) > 0.0):
            warnings.warn(
                f"{label} contains entries outside the structurally allowed SCF support; "
                "those entries were projected away before the first iteration",
                UserWarning,
                stacklevel=3,
            )
            return


def max_norm(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def integration_counters(result: DensityMatrixResult) -> tuple[int, int, int, int, int]:
    info = result.info
    return (
        int(getattr(info, "charge_integration_calls", 0) or 0),
        int(getattr(info, "density_integration_calls", 0) or 0),
        int(getattr(info, "n_kernel_evals", 0) or 0),
        int(
            getattr(
                info,
                "unique_evals",
                getattr(info, "n_kernel_evals", getattr(info, "n_kpoints", 0)),
            )
            or 0
        ),
        int(getattr(info, "n_evaluator_evals", 0) or 0),
    )


def record_density_result(state: SCFRunState, result: DensityMatrixResult) -> None:
    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = integration_counters(
        result
    )
    state.density_matrix_result = result
    state.mu = result.mu
    state.total_charge_integration_calls += charge_calls
    state.total_density_integration_calls += density_calls
    state.total_kernel_evals += kernel_evals
    state.total_unique_evals += unique_evals
    state.total_evaluator_evals += evaluator_evals


def scf_method_name(scf: SCFMethod) -> str:
    if isinstance(scf, AndersonMixing):
        return "anderson_mixing"
    if isinstance(scf, LinearMixing):
        return "linear_mixing"
    return scf.__class__.__name__


def build_scf_info(
    state: SCFRunState,
    *,
    final_result: DensityMatrixResult,
    scf: SCFMethod,
    residual_norm: float,
) -> SCFInfo:
    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = integration_counters(
        final_result
    )
    return SCFInfo(
        method=scf_method_name(scf),
        iterations=max(1, state.iterations),
        residual_norm=residual_norm,
        total_charge_integration_calls=state.total_charge_integration_calls + charge_calls,
        total_density_integration_calls=state.total_density_integration_calls + density_calls,
        total_kernel_evals=state.total_kernel_evals + kernel_evals,
        total_unique_evals=state.total_unique_evals + unique_evals,
        total_evaluator_evals=state.total_evaluator_evals + evaluator_evals,
    )


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
    try:
        from scipy.optimize import anderson
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError("AndersonMixing requires scipy to be installed") from exc

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
    except TypeError as exc:
        if "callback" not in str(exc):
            raise
        try:
            with np.errstate(invalid="ignore"):
                result = anderson(
                    residual_fn,
                    x0,
                    M=int(scf.M),
                    line_search=scf.line_search,
                    maxiter=int(scf.max_iterations),
                    f_tol=scf_tol,
                    tol_norm=max_norm,
                )
        except Exception as inner_exc:  # pragma: no cover - exercised through scipy
            translate_no_convergence(inner_exc, x0)
            raise
        residual = np.asarray(residual_fn(result), dtype=float)
        iterations = max(1, state["iterations"])
        on_iteration(iterations, max_norm(residual))
        return np.asarray(result, dtype=float)
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


def run_scf_problem(
    params0: np.ndarray,
    *,
    evaluate_density: Callable[[np.ndarray, float], DensityMatrixResult],
    residual_from_density: Callable[[np.ndarray, DensityMatrixResult], np.ndarray],
    scf: SCFMethod,
    scf_tol: float,
    state: SCFRunState | None = None,
) -> SCFRunResult:
    run_state = SCFRunState() if state is None else state

    def residual_fn(params: np.ndarray) -> np.ndarray:
        density_result = evaluate_density(params, run_state.mu)
        record_density_result(run_state, density_result)
        residual = np.asarray(residual_from_density(params, density_result), dtype=float)
        run_state.residual_norm = max_norm(residual)
        return residual

    def on_iteration(iteration: int, residual_norm: float) -> None:
        run_state.iterations = iteration
        run_state.residual_norm = residual_norm

    result_params = solve_fixed_point(
        residual_fn,
        params0,
        scf=scf,
        scf_tol=scf_tol,
        on_iteration=on_iteration,
    )
    final_density_result = evaluate_density(result_params, run_state.mu)
    residual_norm = max_norm(
        np.asarray(
            residual_from_density(result_params, final_density_result),
            dtype=float,
        )
    )
    return SCFRunResult(
        params=np.asarray(result_params, dtype=float),
        final_density_result=final_density_result,
        state=run_state,
        residual_norm=residual_norm,
    )


def solve_with_family_adapter(
    model,
    guess: _tb_type,
    *,
    scf: SCFMethod,
    scf_tol: float,
    runtime: SolverRuntime,
    adapter,
) -> SolverResult:
    del model
    if scf_tol <= 0:
        raise ValueError("scf_tol must be positive")

    projected_guess = adapter.project_guess(guess)
    initial_density_result = adapter.evaluate_projected_guess(projected_guess)
    params0 = np.asarray(
        adapter.params_from_density_result(initial_density_result),
        dtype=float,
    )

    state = SCFRunState()
    record_density_result(state, initial_density_result)

    def evaluate_density(params: np.ndarray, mu_guess: float) -> DensityMatrixResult:
        return adapter.evaluate_params(params, mu_guess=mu_guess)

    def residual_from_density(
        params: np.ndarray,
        density_result: DensityMatrixResult,
    ) -> np.ndarray:
        updated = np.asarray(
            adapter.params_from_density_result(density_result),
            dtype=float,
        )
        return updated - np.asarray(params, dtype=float)

    run = run_scf_problem(
        params0,
        evaluate_density=evaluate_density,
        residual_from_density=residual_from_density,
        scf=scf,
        scf_tol=scf_tol,
        state=state,
    )

    density_matrix_result = run.final_density_result
    info = build_scf_info(
        run.state,
        final_result=density_matrix_result,
        scf=scf,
        residual_norm=run.residual_norm,
    )
    return SolverResult(
        mf=adapter.finalize_meanfield(density_matrix_result),
        density_matrix_result=density_matrix_result,
        integration=runtime.integration,
        scf=scf,
        info=info,
    )
