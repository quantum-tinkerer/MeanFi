from __future__ import annotations

from dataclasses import replace

import numpy as np

from meanfi.density.density import evaluate_density
from meanfi.density.problem import build_density_problem
from meanfi.results import DensityResult, DensityEntries
from meanfi.meanfield import bdg_correction_from_density
from meanfi.model import Model
from meanfi.scf.engine import (
    EnergyEvaluation,
    SCFProblem,
    SolverRuntime,
    warn_on_projection,
)
from meanfi.observables import _bdg_correction_expectation
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import _tb_type


def build_bdg_scf_problem(model: Model, runtime: SolverRuntime) -> SCFProblem:
    """Build the superconducting map consumed by the generic SCF engine."""

    space = model.scf_space
    density_problem = build_density_problem(
        model.hamiltonian_from_meanfield(),
        kT=model.kT,
        keys=space.density_keys,
        integration=runtime.integration,
        tolerances=runtime.tolerances,
        density_coordinates=space.required_coordinates,
        electron_ndof=model._ndof,
    )

    def project_guess(guess: _tb_type) -> _tb_type:
        validate_bdg_tb(
            guess,
            ndof=model._ndof,
            ndim=model._ndim,
            name="BdG correction",
        )
        active = space.project_meanfield_input(guess)
        projected = assemble_bdg_tb(
            {key: block[: model._ndof, : model._ndof] for key, block in active.items()},
            {key: block[: model._ndof, model._ndof :] for key, block in active.items()},
            ndof=model._ndof,
        )
        warn_on_projection(guess, projected, label="BdG SCF guess")
        return projected

    def evaluate_meanfield(
        meanfield_guess: _tb_type,
        *,
        mu_guess: float,
    ) -> DensityResult:
        return evaluate_density(
            replace(
                density_problem,
                hamiltonian=model.hamiltonian_from_meanfield(meanfield_guess),
            ),
            filling=model.filling,
            mu_tol=runtime.mu_tol,
            max_charge_evaluations=runtime.max_charge_evaluations,
            mu_guess=mu_guess,
        )

    def evaluate_projected_guess(projected_guess: _tb_type) -> DensityResult:
        return evaluate_meanfield(projected_guess, mu_guess=0.0)

    def state_from_density(density: DensityEntries) -> ActiveDensityState:
        if density.coordinates.entries != space.required_coordinates.entries:
            raise ValueError("density slice does not match the BdG SCF space")
        return ActiveDensityState(
            space,
            space.params_from_required_entries(density.values),
        )

    def active_density(state: ActiveDensityState) -> _tb_type:
        require_same_space(state, space)
        return space.meanfield_input_from_params(state.values)

    def mean_field_from_state(state: ActiveDensityState) -> _tb_type:
        return bdg_correction_from_density(active_density(state), model)

    def evaluate_state(state: ActiveDensityState, mu_guess: float) -> DensityResult:
        return evaluate_meanfield(
            mean_field_from_state(state),
            mu_guess=mu_guess,
        )

    def interaction_energy(params: np.ndarray) -> float:
        state = ActiveDensityState(space, params)
        energy = _bdg_correction_expectation(
            active_density(state), mean_field_from_state(state), model._ndof
        )
        return 0.5 * energy / model._ndof

    def interaction_gradient(params: np.ndarray, direction: np.ndarray) -> float:
        gradient = _bdg_correction_expectation(
            active_density(ActiveDensityState(space, direction)),
            mean_field_from_state(ActiveDensityState(space, params)),
            model._ndof,
        )
        return gradient / model._ndof

    def energy_from_evaluation(input_state, output_state, density):
        correction_energy = _bdg_correction_expectation(
            active_density(output_state),
            mean_field_from_state(input_state),
            model._ndof,
        )
        one_body = density.band_energy - correction_energy / model._ndof
        internal_energy = one_body + interaction_energy(output_state.values)
        return EnergyEvaluation(
            linear_free_energy=one_body - model.kT * density.entropy,
            internal_energy=internal_energy,
            free_energy=internal_energy - model.kT * density.entropy,
        )

    return SCFProblem(
        runtime=runtime,
        kT=model.kT,
        state_space=space,
        project_guess=project_guess,
        evaluate_projected_guess=evaluate_projected_guess,
        state_from_density=state_from_density,
        evaluate_state=evaluate_state,
        mean_field_from_state=mean_field_from_state,
        energy_from_evaluation=energy_from_evaluation,
        interaction_energy=interaction_energy,
        interaction_gradient=interaction_gradient,
    )
