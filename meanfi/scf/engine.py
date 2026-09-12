from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from meanfi.density.integrate.methods import AdaptiveSimplex
from meanfi.density.internal import DensityEvaluation, DensitySlice
from meanfi.errors import ErrorTolerances
from meanfi.results import DensityResult, SCFIteration, SCFResult
from meanfi.scf.ediis import EDIISPoint, ediis_coefficients
from meanfi.scf.fixed_point import (
    NoConvergence,
    SolverError,
    SolverFailure,
    max_norm,
    solve_fixed_point,
)
from meanfi.scf.info import SCFRunState, initialize_run_state, record_scf_iteration
from meanfi.scf.methods import EnergyDIIS, SCFMethod
from meanfi.space.state import ActiveDensityState
from meanfi.tb.ops import _tb_type
from meanfi.tb.storage import tb_entries_changed


@dataclass(frozen=True)
class SolverRuntime:
    integration: object
    tolerances: ErrorTolerances
    mu_tol: float
    max_charge_evaluations: int | None


@dataclass
class SCFRunResult:
    final_state: ActiveDensityState
    final_evaluation: DensityEvaluation
    run_state: SCFRunState
    residual_norm: float
    total_energy: float | None


@dataclass
class _ResidualEvaluation:
    input_state: ActiveDensityState
    output_state: ActiveDensityState
    residual: np.ndarray
    density: DensityEvaluation
    residual_norm: float
    total_energy: float | None


@dataclass(frozen=True)
class EnergyEvaluation:
    output_state: ActiveDensityState
    one_body_energy: float
    total_energy: float


@dataclass(frozen=True)
class SCFProblem:
    runtime: SolverRuntime
    state_space: object
    project_guess: Callable[[_tb_type], _tb_type]
    evaluate_projected_guess: Callable[[_tb_type], DensityEvaluation]
    state_from_density: Callable[[DensitySlice], ActiveDensityState]
    evaluate_state: Callable[[ActiveDensityState, float], DensityEvaluation]
    mean_field_from_state: Callable[[ActiveDensityState], _tb_type]
    energy_from_evaluation: (
        Callable[[ActiveDensityState, DensityEvaluation], EnergyEvaluation | None]
        | None
    ) = None
    interaction_energy: Callable[[np.ndarray], float] | None = None
    interaction_gradient: Callable[[np.ndarray, np.ndarray], float] | None = None


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
    if iteration.total_energy is not None:
        parts.append(f"total_energy={iteration.total_energy:.12g}")
    return " ".join(parts)


def _evaluate_state(
    problem: SCFProblem,
    input_state: ActiveDensityState,
    mu_guess: float,
) -> tuple[DensityEvaluation, ActiveDensityState, EnergyEvaluation | None]:
    density = problem.evaluate_state(input_state, mu_guess)
    output_state = problem.state_from_density(density.density)
    energy = (
        None
        if problem.energy_from_evaluation is None
        else problem.energy_from_evaluation(input_state, density)
    )
    if energy is not None and energy.output_state.space is not output_state.space:
        raise RuntimeError("energy evaluation returned an incompatible SCF state")
    return density, output_state, energy


def iterate_density_fixed_point(
    state0: ActiveDensityState,
    *,
    problem: SCFProblem,
    scf: SCFMethod,
    scf_tol: float,
    verbose: bool = False,
    run_state: SCFRunState,
) -> SCFRunResult:
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
            if _arrays_match(trial.input_state.values, params) and _arrays_match(
                trial.residual, residual
            ):
                return trial_evaluations.pop(index)
        for index in range(len(trial_evaluations) - 1, -1, -1):
            trial = trial_evaluations[index]
            if _arrays_match(trial.input_state.values, params):
                return trial_evaluations.pop(index)
        return None

    def _commit_trial(trial: _ResidualEvaluation) -> SCFIteration:
        iteration = record_scf_iteration(
            run_state,
            trial.density,
            trial.input_state,
            trial.output_state,
            residual_norm=trial.residual_norm,
            total_energy=trial.total_energy,
        )
        trial_evaluations.clear()
        if verbose:
            print(_format_scf_progress(iteration))
        return iteration

    def residual_fn(params: np.ndarray) -> np.ndarray:
        input_state = ActiveDensityState(problem.state_space, params)
        density, output_state, energy = _evaluate_state(
            problem,
            input_state,
            0.0 if run_state.evaluation is None else run_state.evaluation.mu,
        )
        residual = np.asarray(output_state.values - input_state.values, dtype=float)
        trial_evaluations.append(
            _ResidualEvaluation(
                input_state=input_state,
                output_state=output_state,
                residual=np.array(residual, copy=True),
                density=density,
                residual_norm=max_norm(residual),
                total_energy=None if energy is None else energy.total_energy,
            )
        )
        return residual

    def on_iteration(
        iteration: int | None,
        residual_norm: float,
        params: np.ndarray,
        residual: np.ndarray,
    ) -> None:
        del iteration, residual_norm
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
        and run_state.evaluation is not None
        and run_state.output_state is not None
        and run_state.residual_norm is not None
    ):
        return SCFRunResult(
            final_state=run_state.output_state,
            final_evaluation=run_state.evaluation,
            run_state=run_state,
            residual_norm=run_state.residual_norm,
            total_energy=run_state.total_energy,
        )

    input_state = ActiveDensityState(problem.state_space, result_params)
    density, output_state, energy = _evaluate_state(
        problem,
        input_state,
        0.0 if run_state.evaluation is None else run_state.evaluation.mu,
    )
    residual = np.asarray(output_state.values - input_state.values, dtype=float)
    final_trial = _ResidualEvaluation(
        input_state=input_state,
        output_state=output_state,
        residual=residual,
        density=density,
        residual_norm=max_norm(residual),
        total_energy=None if energy is None else energy.total_energy,
    )
    _commit_trial(final_trial)
    return SCFRunResult(
        final_state=output_state,
        final_evaluation=density,
        run_state=run_state,
        residual_norm=final_trial.residual_norm,
        total_energy=final_trial.total_energy,
    )


def iterate_energy_ediis(
    state0: ActiveDensityState,
    *,
    problem: SCFProblem,
    scf: EnergyDIIS,
    verbose: bool,
    run_state: SCFRunState,
) -> SCFRunResult:
    if not isinstance(problem.runtime.integration, AdaptiveSimplex):
        raise ValueError("EnergyDIIS requires AdaptiveSimplex integration")
    if (
        problem.energy_from_evaluation is None
        or problem.interaction_energy is None
        or problem.interaction_gradient is None
    ):
        raise ValueError("EnergyDIIS requires a normal-state energy functional")

    scf_tol = problem.runtime.tolerances.scf_residual
    params = np.array(state0.values, copy=True)
    history: list[EDIISPoint] = []

    for _iteration in range(1, int(scf.max_iterations) + 1):
        input_state = ActiveDensityState(problem.state_space, params)
        density, output_state, energy = _evaluate_state(
            problem,
            input_state,
            0.0 if run_state.evaluation is None else run_state.evaluation.mu,
        )
        if energy is None:
            raise RuntimeError("EnergyDIIS requires an occupied band-energy result")
        residual = np.asarray(output_state.values - input_state.values, dtype=float)
        residual_norm = max_norm(residual)
        iteration = record_scf_iteration(
            run_state,
            density,
            input_state,
            output_state,
            residual_norm=residual_norm,
            total_energy=energy.total_energy,
        )
        if verbose:
            print(_format_scf_progress(iteration))
        if residual_norm <= scf_tol:
            return SCFRunResult(
                final_state=output_state,
                final_evaluation=density,
                run_state=run_state,
                residual_norm=residual_norm,
                total_energy=energy.total_energy,
            )

        history.append(
            EDIISPoint(
                params=np.array(output_state.values, copy=True),
                one_body_energy=energy.one_body_energy,
                energy=energy.total_energy,
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
    if state.evaluation is None or state.output_state is None:
        raise RuntimeError("cannot build an SCF result before a successful evaluation")
    errors = state.evaluation.errors
    if state.residual_norm is not None:
        errors = replace(errors, scf_residual=state.residual_norm)
    density = DensityResult(
        coordinates=state.evaluation.density.coordinates,
        values=state.evaluation.density.values,
        mu=float(state.evaluation.mu),
        filling=float(state.evaluation.filling),
        errors=errors,
        statistics=state.evaluation.statistics,
    )
    return SCFResult(
        density=density,
        mean_field=problem.mean_field_from_state(state.output_state),
        total_energy=state.total_energy,
        errors=errors,
        history=tuple(state.history or ()),
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
    initial_density = problem.evaluate_projected_guess(projected_guess)
    initial_state = problem.state_from_density(initial_density.density)
    run_state = SCFRunState()
    initialize_run_state(run_state, initial_density, initial_state)

    try:
        if isinstance(scf, EnergyDIIS):
            run = iterate_energy_ediis(
                initial_state,
                problem=problem,
                scf=scf,
                verbose=verbose,
                run_state=run_state,
            )
        else:
            run = iterate_density_fixed_point(
                initial_state,
                problem=problem,
                scf=scf,
                scf_tol=problem.runtime.tolerances.scf_residual,
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

    result = _build_result(problem, run.run_state, converged=True)
    if result.errors.scf_residual is None:
        raise RuntimeError("converged SCF result is missing its residual")
    return result


__all__ = [
    "EnergyEvaluation",
    "NoConvergence",
    "SCFProblem",
    "SolverError",
    "SolverFailure",
    "SolverRuntime",
    "iterate_density_fixed_point",
    "iterate_energy_ediis",
    "run_scf_loop",
    "warn_on_projection",
]
