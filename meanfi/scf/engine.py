from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from meanfi.results import DensityMatrixResult, SCFIterationInfo, SolverResult
from meanfi.scf.fixed_point import NoConvergence, max_norm, solve_fixed_point
from meanfi.scf.info import (
    SCFRunState,
    accept_density_result,
    build_scf_info,
    record_density_evaluation,
    record_density_result,
    record_scf_iteration,
)
from meanfi.scf.methods import SCFMethod
from meanfi.tb.ops import _tb_type
from meanfi.tb.storage import tb_entries_changed


@dataclass(frozen=True)
class SolverRuntime:
    integration: object
    filling_tol: float | None
    mu_tol: float
    max_charge_evaluations: int | None


@dataclass
class SCFRunResult:
    params: np.ndarray
    final_density_result: DensityMatrixResult
    state: SCFRunState
    residual_norm: float


@dataclass
class _ResidualEvaluation:
    params: np.ndarray
    residual: np.ndarray
    density_result: DensityMatrixResult
    residual_norm: float
    line_search_norm: float


@dataclass(frozen=True)
class SCFProblem:
    runtime: SolverRuntime
    project_guess: Callable[[_tb_type], _tb_type]
    evaluate_projected_guess: Callable[[_tb_type], DensityMatrixResult]
    compress_density: Callable[[_tb_type], np.ndarray]
    density_result_from_params: Callable[[np.ndarray, float], DensityMatrixResult]
    finalize_meanfield: Callable[[DensityMatrixResult], _tb_type]


def warn_on_projection(original: _tb_type, projected: _tb_type, *, label: str) -> None:
    import warnings

    if not tb_entries_changed(original, projected):
        return
    warnings.warn(
        f"{label} contains values outside the active SCF density selection; "
        "those values were projected away before the first iteration",
        UserWarning,
        stacklevel=3,
    )


def _format_scf_progress(info: SCFIterationInfo) -> str:
    parts = [
        f"scf step={info.step}",
        f"residual={info.residual_norm:.6e}",
        f"line_search_norm={info.line_search_norm:.6e}",
        f"integration_evals={info.integration_evals}",
        f"cumulative_integration_evals={info.cumulative_integration_evals}",
        f"mu={info.mu:.12g}",
    ]
    if info.filling_residual is not None:
        parts.append(f"filling_residual={info.filling_residual:.6e}")
    if info.charge_error is not None:
        parts.append(f"charge_error={info.charge_error:.6e}")
    return " ".join(parts)


def iterate_density_fixed_point(
    params0: np.ndarray,
    *,
    density_result_from_params: Callable[[np.ndarray, float], DensityMatrixResult],
    compress_density: Callable[[_tb_type], np.ndarray],
    scf: SCFMethod,
    scf_tol: float,
    verbose: bool = False,
    state: SCFRunState | None = None,
) -> SCFRunResult:
    run_state = SCFRunState() if state is None else state
    trial_evaluations: list[_ResidualEvaluation] = []

    def _arrays_match(left: np.ndarray, right: np.ndarray) -> bool:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if left_array.shape == right_array.shape:
            return bool(np.array_equal(left_array, right_array))
        return bool(np.array_equal(np.ravel(left_array), np.ravel(right_array)))

    def _pop_trial(
        params: np.ndarray, residual: np.ndarray
    ) -> _ResidualEvaluation | None:
        for index in range(len(trial_evaluations) - 1, -1, -1):
            trial = trial_evaluations[index]
            if _arrays_match(trial.params, params) and _arrays_match(
                trial.residual, residual
            ):
                return trial_evaluations.pop(index)
        for index in range(len(trial_evaluations) - 1, -1, -1):
            trial = trial_evaluations[index]
            if _arrays_match(trial.params, params):
                return trial_evaluations.pop(index)
        return None

    def _commit_trial(
        trial: _ResidualEvaluation, *, iteration: int | None
    ) -> SCFIterationInfo:
        if iteration is not None:
            run_state.iterations = iteration
        run_state.residual_norm = trial.residual_norm
        accept_density_result(run_state, trial.density_result)
        iteration_info = record_scf_iteration(
            run_state,
            trial.density_result,
            residual_norm=trial.residual_norm,
            line_search_norm=trial.line_search_norm,
        )
        trial_evaluations.clear()
        if verbose:
            print(_format_scf_progress(iteration_info))
        return iteration_info

    def residual_fn(params: np.ndarray) -> np.ndarray:
        density_result = density_result_from_params(params, run_state.mu)
        record_density_evaluation(run_state, density_result)
        updated = np.asarray(
            compress_density(density_result.density_matrix), dtype=float
        )
        residual = updated - np.asarray(params, dtype=float)
        residual_norm = max_norm(residual)
        line_search_norm = float(np.linalg.norm(np.ravel(residual)))
        trial_evaluations.append(
            _ResidualEvaluation(
                params=np.array(params, dtype=float, copy=True),
                residual=np.array(residual, dtype=float, copy=True),
                density_result=density_result,
                residual_norm=residual_norm,
                line_search_norm=line_search_norm,
            )
        )
        return residual

    def on_iteration(
        iteration: int | None,
        residual_norm: float,
        params: np.ndarray,
        residual: np.ndarray,
    ) -> None:
        del residual_norm
        trial = _pop_trial(params, residual)
        if trial is None:
            residual = residual_fn(params)
            trial = _pop_trial(params, residual)
        if trial is None:  # pragma: no cover - defensive fallback
            raise RuntimeError("accepted SCF iterate was not evaluated")
        _commit_trial(trial, iteration=iteration)

    result_params = solve_fixed_point(
        residual_fn,
        params0,
        scf=scf,
        scf_tol=scf_tol,
        on_iteration=on_iteration,
    )
    final_density_result = density_result_from_params(result_params, run_state.mu)
    final_residual = np.asarray(
        compress_density(final_density_result.density_matrix)
        - np.asarray(result_params, dtype=float),
        dtype=float,
    )
    residual_norm = max_norm(final_residual)
    line_search_norm = float(np.linalg.norm(np.ravel(final_residual)))
    accept_density_result(run_state, final_density_result)
    final_iteration_info = record_scf_iteration(
        run_state,
        final_density_result,
        residual_norm=residual_norm,
        line_search_norm=line_search_norm,
    )
    if verbose:
        print(_format_scf_progress(final_iteration_info))
    return SCFRunResult(
        params=np.asarray(result_params, dtype=float),
        final_density_result=final_density_result,
        state=run_state,
        residual_norm=residual_norm,
    )


def run_scf_loop(
    guess: _tb_type,
    *,
    scf: SCFMethod,
    scf_tol: float,
    problem: SCFProblem,
    verbose: bool = False,
) -> SolverResult:
    if scf_tol <= 0:
        raise ValueError("scf_tol must be positive")

    projected_guess = problem.project_guess(guess)
    initial_density_result = problem.evaluate_projected_guess(projected_guess)
    params0 = np.asarray(
        problem.compress_density(initial_density_result.density_matrix),
        dtype=float,
    )

    state = SCFRunState()
    record_density_result(state, initial_density_result)

    run = iterate_density_fixed_point(
        params0,
        density_result_from_params=problem.density_result_from_params,
        compress_density=problem.compress_density,
        scf=scf,
        scf_tol=scf_tol,
        verbose=verbose,
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
        mf=problem.finalize_meanfield(density_matrix_result),
        density_matrix_result=density_matrix_result,
        integration=problem.runtime.integration,
        scf=scf,
        info=info,
    )


__all__ = [
    "NoConvergence",
    "SCFProblem",
    "SolverRuntime",
    "iterate_density_fixed_point",
    "run_scf_loop",
    "warn_on_projection",
]
