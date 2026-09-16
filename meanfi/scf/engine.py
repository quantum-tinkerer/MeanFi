"""Self-consistency iteration and accepted physical results."""

from dataclasses import replace
import numpy as np

from meanfi.errors import ConvergenceError
from meanfi.results import SCFIteration, SCFResult
from meanfi.observables import _internal_energy_from_band
from meanfi.scf.ediis import EDIISPoint, ediis_coefficients
from meanfi.scf.fixed_point import (
    NoConvergence,
    SolverError,
    SolverFailure,
    iterate_anderson,
)
from meanfi.scf.methods import AndersonMixing, EnergyDIIS, LinearMixing, SCFMethod
from meanfi.scf.problem import SCFEvaluation, SCFProblem
from meanfi.space.state import ActiveDensityState
from meanfi.tb.ops import _tb_type


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
    parts.append(f"internal_energy={iteration.internal_energy:.12g}")
    return " ".join(parts)


def _build_result(problem, evaluation, history, *, converged, compute_free_energy):
    errors = replace(evaluation.density.errors, scf_residual=evaluation.residual_norm)
    density = replace(
        evaluation.density, errors=errors, internal_energy=evaluation.internal_energy
    )
    result = SCFResult(
        density=density,
        mean_field=problem.model._mean_field_from_state(evaluation.output_state),
        history=tuple(history),
        converged=converged,
    )
    if compute_free_energy:
        try:
            thermal = problem.evaluate_mean_field(
                evaluation.mean_field,
                mu_guess=density.mu,
                mu=density.mu,
                compute_entropy=True,
            )
        except (ConvergenceError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise SolverFailure(
                "Final entropy calculation failed", result=result
            ) from exc
        density = replace(
            density,
            entropy=thermal.entropy,
            errors=replace(errors, entropy=thermal.errors.entropy),
        )
        result = replace(result, density=density)
    return result


def run_scf_loop(
    guess: _tb_type,
    *,
    scf: SCFMethod,
    problem: SCFProblem,
    verbose: bool = False,
    compute_free_energy: bool = True,
) -> SCFResult:
    projected_guess = problem.project_guess(guess)
    try:
        density = problem.evaluate_mean_field(projected_guess, mu_guess=0.0)
    except (ConvergenceError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise SolverFailure("Initial SCF density evaluation failed") from exc
    state = problem.state_from_density(density.entries)
    energy = _internal_energy_from_band(
        problem.model, state, density.band_energy, projected_guess
    )
    last = SCFEvaluation(
        density, state, internal_energy=energy, mean_field=projected_guess
    )
    history: list[SCFIteration] = []
    tolerance = problem.density_problem.tolerances.scf_residual

    def evaluate(params):
        state = ActiveDensityState(problem.model._space, params)
        return problem.evaluate_state(state, last.density.mu)

    def accept(evaluation):
        nonlocal last
        last = evaluation
        iteration = SCFIteration(
            step=len(history) + 1,
            mu=last.density.mu,
            filling=last.density.filling,
            internal_energy=last.internal_energy,
            errors=replace(last.density.errors, scf_residual=last.residual_norm),
        )
        history.append(iteration)
        if verbose:
            print(_format_scf_progress(iteration))

    params = last.output_state.values
    try:
        if isinstance(scf, AndersonMixing):
            iterate_anderson(
                evaluate,
                params,
                scf=scf,
                scf_tol=tolerance,
                accept=accept,
                residual_norm=problem.model._space.density_norm,
            )
        else:
            points: list[EDIISPoint] = []
            for _ in range(scf.max_iterations):
                evaluation = evaluate(params)
                accept(evaluation)
                if evaluation.residual_norm <= tolerance:
                    break
                if isinstance(scf, LinearMixing):
                    params = params + scf.alpha * evaluation.residual
                elif isinstance(scf, EnergyDIIS):
                    points.append(
                        EDIISPoint(
                            evaluation.output_state.values, evaluation.internal_energy
                        )
                    )
                    points = points[-scf.history_size :]
                    weights = ediis_coefficients(
                        points, interaction_curvature=problem.interaction_curvature
                    )
                    params = weights @ np.stack([point.params for point in points])
                else:
                    raise TypeError(
                        "scf must be LinearMixing, EnergyDIIS, or AndersonMixing"
                    )
            else:
                raise NoConvergence(params)
    except NoConvergence as exc:
        partial = _build_result(
            problem,
            last,
            history,
            converged=False,
            compute_free_energy=compute_free_energy,
        )
        raise NoConvergence(exc.last_iterate, result=partial) from exc
    except SolverError:
        raise
    except Exception as exc:
        partial = _build_result(
            problem,
            last,
            history,
            converged=False,
            compute_free_energy=compute_free_energy,
        )
        raise SolverFailure("SCF evaluation failed", result=partial) from exc

    result = _build_result(
        problem, last, history, converged=True, compute_free_energy=compute_free_energy
    )
    if result.errors.scf_residual is None or result.errors.scf_residual > tolerance:
        raise NoConvergence(
            last.output_state.values, result=replace(result, converged=False)
        )
    if verbose and result.entropy is not None:
        print(
            f"scf converged: entropy={result.entropy:.12g} free_energy={result.free_energy:.12g}"
        )
    return result
