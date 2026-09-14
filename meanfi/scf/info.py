from __future__ import annotations

from dataclasses import dataclass, field, replace

from meanfi.results import DensityResult
from meanfi.results import SCFIteration
from meanfi.space.state import ActiveDensityState


@dataclass
class SCFRunState:
    """Last successful SCF evaluation plus its compact public history."""

    evaluation: DensityResult
    output_state: ActiveDensityState
    input_state: ActiveDensityState | None = None
    residual_norm: float | None = None
    internal_energy: float | None = None
    free_energy: float | None = None
    history: list[SCFIteration] = field(default_factory=list)


def record_scf_iteration(
    state: SCFRunState,
    evaluation: DensityResult,
    input_state: ActiveDensityState,
    output_state: ActiveDensityState,
    *,
    residual_norm: float,
    internal_energy: float | None,
    free_energy: float | None,
) -> SCFIteration:
    errors = replace(evaluation.errors, scf_residual=float(residual_norm))
    iteration = SCFIteration(
        step=len(state.history) + 1,
        mu=float(evaluation.mu),
        filling=float(evaluation.filling),
        internal_energy=internal_energy,
        free_energy=free_energy,
        entropy=evaluation.entropy,
        errors=errors,
    )
    state.history.append(iteration)
    state.evaluation = evaluation
    state.input_state = input_state
    state.output_state = output_state
    state.residual_norm = float(residual_norm)
    state.internal_energy = internal_energy
    state.free_energy = free_energy
    return iteration
