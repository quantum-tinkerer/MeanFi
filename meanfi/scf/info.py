from __future__ import annotations

from dataclasses import dataclass, replace

from meanfi.density.internal import DensityEvaluation
from meanfi.results import SCFIteration
from meanfi.space.state import ActiveDensityState


@dataclass
class SCFRunState:
    """Last successful SCF evaluation plus its compact public history."""

    evaluation: DensityEvaluation | None = None
    input_state: ActiveDensityState | None = None
    output_state: ActiveDensityState | None = None
    residual_norm: float | None = None
    total_energy: float | None = None
    history: list[SCFIteration] | None = None


def initialize_run_state(
    state: SCFRunState,
    evaluation: DensityEvaluation,
    output_state: ActiveDensityState,
) -> None:
    state.evaluation = evaluation
    state.output_state = output_state


def record_scf_iteration(
    state: SCFRunState,
    evaluation: DensityEvaluation,
    input_state: ActiveDensityState,
    output_state: ActiveDensityState,
    *,
    residual_norm: float,
    total_energy: float | None,
) -> SCFIteration:
    errors = replace(evaluation.errors, scf_residual=float(residual_norm))
    history = [] if state.history is None else state.history
    iteration = SCFIteration(
        step=len(history) + 1,
        mu=float(evaluation.mu),
        filling=float(evaluation.filling),
        total_energy=None if total_energy is None else float(total_energy),
        errors=errors,
    )
    history.append(iteration)
    state.evaluation = evaluation
    state.input_state = input_state
    state.output_state = output_state
    state.residual_norm = float(residual_norm)
    state.total_energy = None if total_energy is None else float(total_energy)
    state.history = history
    return iteration
