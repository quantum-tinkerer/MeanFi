from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from meanfi.errors import ConvergenceError
from meanfi.results import DensityResult, SCFIteration, SCFResult
from meanfi.scf.ediis import EDIISPoint, ediis_coefficients
from meanfi.scf.fixed_point import (
    NoConvergence,
    SolverError,
    SolverFailure,
    max_norm,
    solve_fixed_point,
)
from meanfi.scf.info import SCFRunState, record_scf_iteration
from meanfi.scf.methods import EnergyDIIS, SCFMethod
from meanfi.scf.problem import EnergyEvaluation, SCFProblem
from meanfi.space.state import ActiveDensityState
from meanfi.tb.ops import _tb_type


@dataclass
class _ResidualEvaluation:
    input_state: ActiveDensityState
    output_state: ActiveDensityState
    residual: np.ndarray
    density: DensityResult
    residual_norm: float
    energy: EnergyEvaluation


def _format_scf_progress(iteration: SCFIteration) -> str:
    parts = [
        f"scf step={iteration.step}",
        f"residual={iteration.errors.scf_residual:.6e}",
        f"mu={iteration.mu:.12g}",
        f"filling={iteration.filling:.12g}",
    ]
    if iteration.errors.filling_residual is not None:
        parts.append(f"filling_residual={iteration.errors.filling_residual:.6e}")
    if iteration.errors.charge_integration is not None:
        parts.append(f"charge_error={iteration.errors.charge_integration:.6e}")
    parts.append(f"free_energy={iteration.free_energy:.12g}")
    return " ".join(parts)


def iterate_density_fixed_point(
    state0: ActiveDensityState,
    *,
    problem: SCFProblem,
    scf: SCFMethod,
    verbose: bool = False,
    run_state: SCFRunState,
) -> None:
    scf_tol = problem.density_problem.tolerances.scf_residual
    trial_evaluations: list[_ResidualEvaluation] = []

    def _pop_trial(
        params: np.ndarray, residual: np.ndarray
    ) -> _ResidualEvaluation | None:
        for index in range(len(trial_evaluations) - 1, -1, -1):
            trial = trial_evaluations[index]
            if np.array_equal(
                trial.input_state.values, np.ravel(params)
            ) and np.array_equal(trial.residual, np.ravel(residual)):
                return trial_evaluations.pop(index)
        for index in range(len(trial_evaluations) - 1, -1, -1):
            trial = trial_evaluations[index]
            if np.array_equal(trial.input_state.values, np.ravel(params)):
                return trial_evaluations.pop(index)
        return None

    def _commit_trial(trial: _ResidualEvaluation) -> SCFIteration:
        iteration = record_scf_iteration(
            run_state,
            trial.density,
            trial.input_state,
            trial.output_state,
            residual_norm=trial.residual_norm,
            internal_energy=trial.energy.internal_energy,
            free_energy=trial.energy.free_energy,
        )
        trial_evaluations.clear()
        if verbose:
            print(_format_scf_progress(iteration))
        return iteration

    def residual_fn(params: np.ndarray) -> np.ndarray:
        input_state = ActiveDensityState(problem.model.scf_space, params)
        density, output_state, energy = problem.evaluate_state(
            input_state,
            run_state.evaluation.mu,
        )
        residual = np.asarray(output_state.values - input_state.values, dtype=float)
        trial_evaluations.append(
            _ResidualEvaluation(
                input_state=input_state,
                output_state=output_state,
                residual=np.array(residual, copy=True),
                density=density,
                residual_norm=max_norm(residual),
                energy=energy,
            )
        )
        return residual

    def on_iteration(
        params: np.ndarray,
        residual: np.ndarray,
    ) -> None:
        trial = _pop_trial(params, residual)
        if trial is None:
            residual = residual_fn(params)
            trial = _pop_trial(params, residual)
        if trial is None:  # pragma: no cover - defensive fallback
            raise RuntimeError("accepted SCF iterate was not evaluated")
        _commit_trial(trial)

    result_params = solve_fixed_point(
        residual_fn,
        state0.values,
        scf=scf,
        scf_tol=scf_tol,
        on_iteration=on_iteration,
    )
    if (
        run_state.input_state is not None
        and np.array_equal(result_params, run_state.input_state.values)
        and run_state.residual_norm is not None
    ):
        return

    input_state = ActiveDensityState(problem.model.scf_space, result_params)
    density, output_state, energy = problem.evaluate_state(
        input_state,
        run_state.evaluation.mu,
    )
    residual = np.asarray(output_state.values - input_state.values, dtype=float)
    final_trial = _ResidualEvaluation(
        input_state=input_state,
        output_state=output_state,
        residual=residual,
        density=density,
        residual_norm=max_norm(residual),
        energy=energy,
    )
    _commit_trial(final_trial)


def iterate_energy_ediis(
    state0: ActiveDensityState,
    *,
    problem: SCFProblem,
    scf: EnergyDIIS,
    verbose: bool,
    run_state: SCFRunState,
) -> None:
    scf_tol = problem.density_problem.tolerances.scf_residual
    params = np.array(state0.values, copy=True)
    history: list[EDIISPoint] = []

    for _ in range(scf.max_iterations):
        input_state = ActiveDensityState(problem.model.scf_space, params)
        density, output_state, energy = problem.evaluate_state(
            input_state,
            run_state.evaluation.mu,
        )
        residual = np.asarray(output_state.values - input_state.values, dtype=float)
        residual_norm = max_norm(residual)
        iteration = record_scf_iteration(
            run_state,
            density,
            input_state,
            output_state,
            residual_norm=residual_norm,
            internal_energy=energy.internal_energy,
            free_energy=energy.free_energy,
        )
        if verbose:
            print(_format_scf_progress(iteration))
        if residual_norm <= scf_tol:
            return

        history.append(
            EDIISPoint(
                params=np.array(output_state.values, copy=True),
                linear_free_energy=energy.linear_free_energy,
                free_energy=energy.free_energy,
            )
        )
        if len(history) > scf.history_size:
            history.pop(0)
        coefficients = ediis_coefficients(
            history,
            interaction_energy=problem.interaction_energy,
            interaction_gradient=problem.interaction_gradient,
        )
        params = np.tensordot(
            coefficients,
            np.stack([point.params for point in history]),
            axes=1,
        )

    raise NoConvergence(params)


def _build_result(
    problem: SCFProblem,
    state: SCFRunState,
    *,
    converged: bool,
) -> SCFResult:
    errors = state.evaluation.errors
    if state.residual_norm is not None:
        errors = replace(errors, scf_residual=state.residual_norm)
    density = replace(state.evaluation, errors=errors)
    return SCFResult(
        density=density,
        mean_field=problem.model._mean_field_from_state(state.output_state),
        internal_energy=state.internal_energy,
        free_energy=state.free_energy,
        errors=errors,
        history=tuple(state.history),
        converged=converged,
    )


def run_scf_loop(
    guess: _tb_type,
    *,
    scf: SCFMethod,
    problem: SCFProblem,
    verbose: bool = False,
) -> SCFResult:
    projected_guess = problem.project_guess(guess)
    try:
        initial_density = problem.evaluate_mean_field(projected_guess, mu_guess=0.0)
    except (ConvergenceError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise SolverFailure("Initial SCF density evaluation failed") from exc
    initial_state = problem.state_from_density(initial_density.entries)
    run_state = SCFRunState(evaluation=initial_density, output_state=initial_state)

    try:
        if isinstance(scf, EnergyDIIS):
            iterate_energy_ediis(
                initial_state,
                problem=problem,
                scf=scf,
                verbose=verbose,
                run_state=run_state,
            )
        else:
            iterate_density_fixed_point(
                initial_state,
                problem=problem,
                scf=scf,
                verbose=verbose,
                run_state=run_state,
            )
    except NoConvergence as exc:
        partial = _build_result(problem, run_state, converged=False)
        raise NoConvergence(exc.last_iterate, result=partial) from exc
    except SolverError:
        raise
    except Exception as exc:
        partial = _build_result(problem, run_state, converged=False)
        raise SolverFailure("SCF evaluation failed", result=partial) from exc

    result = _build_result(problem, run_state, converged=True)
    if result.errors.scf_residual is None:
        raise RuntimeError("converged SCF result is missing its residual")
    return result


__all__ = [
    "NoConvergence",
    "SolverError",
    "SolverFailure",
    "iterate_density_fixed_point",
    "iterate_energy_ediis",
    "run_scf_loop",
]
