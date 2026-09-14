from __future__ import annotations

from dataclasses import replace

import numpy as np

from meanfi.density.density import evaluate_density
from meanfi.density.problem import build_normal_problem
from meanfi.density.integrate.methods import AdaptiveSimplex
from meanfi.results import DensityResult, DensityEntries
from meanfi.meanfield import meanfield
from meanfi.model import Model
from meanfi.observables import expectation_value
from meanfi.scf.engine import (
    EnergyEvaluation,
    SCFProblem,
    SolverRuntime,
    warn_on_projection,
)
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.tb.ops import _tb_type


def build_normal_scf_problem(model: Model, runtime: SolverRuntime) -> SCFProblem:
    """Build the normal-state map consumed by the generic SCF engine."""

    space = model.scf_space
    keys = space.density_keys
    density_problem = build_normal_problem(
        model.h_0,
        kT=model.kT,
        keys=keys,
        integration=runtime.integration,
        tolerances=runtime.tolerances,
        density_coordinates=space.required_coordinates,
        include_band_energy=isinstance(runtime.integration, AdaptiveSimplex),
    )

    def project_guess(guess: _tb_type) -> _tb_type:
        projected = space.project_meanfield_input(guess)
        warn_on_projection(guess, projected, label="Normal SCF guess")
        return projected

    def evaluate_hamiltonian(
        hamiltonian: _tb_type,
        *,
        mu_guess: float,
    ) -> DensityResult:
        return evaluate_density(
            replace(density_problem, hamiltonian=hamiltonian),
            filling=model.filling,
            mu_tol=runtime.mu_tol,
            max_charge_evaluations=runtime.max_charge_evaluations,
            mu_guess=mu_guess,
        )

    def evaluate_projected_guess(projected_guess: _tb_type) -> DensityResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_meanfield(projected_guess),
            mu_guess=0.0,
        )

    def state_from_density(density: DensityEntries) -> ActiveDensityState:
        if density.coordinates.entries != space.required_coordinates.entries:
            raise ValueError("density slice does not match the normal SCF space")
        return ActiveDensityState(
            space,
            space.params_from_required_entries(density.values),
        )

    def active_density(state: ActiveDensityState) -> _tb_type:
        require_same_space(state, space)
        return space.meanfield_input_from_params(state.values)

    def evaluate_state(state: ActiveDensityState, mu_guess: float) -> DensityResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_rho(active_density(state)),
            mu_guess=mu_guess,
        )

    def density_difference(state: ActiveDensityState) -> _tb_type:
        return model._active_density_from_state(model._reference_difference(state))

    def interaction_energy_values(params: np.ndarray) -> float:
        state = ActiveDensityState(space, params)
        difference = density_difference(state)
        correction = meanfield(difference, model.h_int)
        return float(0.5 * np.real(expectation_value(difference, correction)))

    def interaction_gradient(
        params: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        state = ActiveDensityState(space, params)
        correction = meanfield(density_difference(state), model.h_int)
        direction_density = active_density(ActiveDensityState(space, direction))
        return float(np.real(expectation_value(direction_density, correction)))

    def energy_from_evaluation(
        input_state: ActiveDensityState,
        output_state: ActiveDensityState,
        density: DensityResult,
    ) -> EnergyEvaluation | None:
        if density.band_energy is None:
            return None
        output_density = active_density(output_state)
        input_correction = meanfield(density_difference(input_state), model.h_int)
        one_body = float(
            density.band_energy
            - np.real(expectation_value(output_density, input_correction))
        )
        total_energy = one_body + interaction_energy_values(output_state.values)
        return EnergyEvaluation(
            one_body_energy=one_body,
            total_energy=total_energy,
        )

    def mean_field_from_state(state: ActiveDensityState) -> _tb_type:
        difference = density_difference(state)
        return dict(meanfield(difference, model.h_int))

    return SCFProblem(
        runtime=runtime,
        state_space=space,
        project_guess=project_guess,
        evaluate_projected_guess=evaluate_projected_guess,
        state_from_density=state_from_density,
        evaluate_state=evaluate_state,
        mean_field_from_state=mean_field_from_state,
        energy_from_evaluation=energy_from_evaluation,
        interaction_energy=interaction_energy_values,
        interaction_gradient=interaction_gradient,
    )
