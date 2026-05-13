from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from meanfi.results import DensityMatrixResult, SolverResult
from meanfi.scf.fixed_point import NoConvergence, max_norm, solve_fixed_point
from meanfi.scf.info import SCFRunState, build_scf_info, record_density_result
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


def iterate_density_fixed_point(
    params0: np.ndarray,
    *,
    density_result_from_params: Callable[[np.ndarray, float], DensityMatrixResult],
    compress_density: Callable[[_tb_type], np.ndarray],
    scf: SCFMethod,
    scf_tol: float,
    state: SCFRunState | None = None,
) -> SCFRunResult:
    run_state = SCFRunState() if state is None else state

    def residual_fn(params: np.ndarray) -> np.ndarray:
        density_result = density_result_from_params(params, run_state.mu)
        record_density_result(run_state, density_result)
        updated = np.asarray(
            compress_density(density_result.density_matrix), dtype=float
        )
        residual = updated - np.asarray(params, dtype=float)
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
    final_density_result = density_result_from_params(result_params, run_state.mu)
    residual_norm = max_norm(
        np.asarray(
            compress_density(final_density_result.density_matrix)
            - np.asarray(result_params, dtype=float),
            dtype=float,
        )
    )
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
