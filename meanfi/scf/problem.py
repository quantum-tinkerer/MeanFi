"""Shared normal/BdG SCF evaluation and its physical energy functional."""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np

from meanfi.density.density import evaluate_density
from meanfi.density.problem import DensityProblem
from meanfi.model import Model
from meanfi.observables import _bdg_correction_expectation, expectation_value
from meanfi.results import DensityEntries, DensityResult
from meanfi.space.state import ActiveDensityState
from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import _tb_type
from meanfi.tb.storage import tb_entries_changed


@dataclass(frozen=True)
class EnergyEvaluation:
    linear_free_energy: float
    internal_energy: float
    free_energy: float


@dataclass(frozen=True)
class SCFProblem:
    model: Model
    density_problem: DensityProblem
    mu_tol: float = 1e-10
    max_charge_evaluations: int | None = None

    def project_guess(self, guess: _tb_type) -> _tb_type:
        model = self.model
        if model.superconducting:
            validate_bdg_tb(
                guess, ndof=model._ndof, ndim=model._ndim, name="BdG correction"
            )
        projected = model.scf_space.project_meanfield_input(guess)
        if model.superconducting:
            projected = assemble_bdg_tb(
                {
                    key: block[: model._ndof, : model._ndof]
                    for key, block in projected.items()
                },
                {
                    key: block[: model._ndof, model._ndof :]
                    for key, block in projected.items()
                },
                ndof=model._ndof,
            )
        if tb_entries_changed(guess, projected):
            warnings.warn(
                "SCF guess contains values outside the active SCF density selection; "
                "those values were projected away before the first iteration",
                UserWarning,
                stacklevel=3,
            )
        return projected

    def evaluate_mean_field(
        self, mean_field: _tb_type, mu_guess: float
    ) -> DensityResult:
        return evaluate_density(
            replace(
                self.density_problem,
                hamiltonian=self.model.hamiltonian_from_meanfield(mean_field),
            ),
            filling=self.model.filling,
            mu_tol=self.mu_tol,
            max_charge_evaluations=self.max_charge_evaluations,
            mu_guess=mu_guess,
        )

    def state_from_density(self, density: DensityEntries) -> ActiveDensityState:
        space = self.model.scf_space
        if density.coordinates.entries != space.required_coordinates.entries:
            raise ValueError("density slice does not match the SCF space")
        return ActiveDensityState(
            space, space.params_from_required_entries(density.values)
        )

    def correction_expectation(self, density: _tb_type, correction: _tb_type) -> float:
        """Correction energy per physical orbital, without Nambu double counting."""
        if self.model.superconducting:
            energy = _bdg_correction_expectation(density, correction, self.model._ndof)
        else:
            energy = float(np.real(expectation_value(density, correction)))
        return energy / self.model._ndof

    def interaction_energy(self, params: np.ndarray) -> float:
        state = ActiveDensityState(self.model.scf_space, params)
        difference = self.model._active_density_from_state(
            self.model._reference_difference(state)
        )
        correction = self.model._mean_field_from_state(state)
        return 0.5 * self.correction_expectation(difference, correction)

    def interaction_gradient(self, params: np.ndarray, direction: np.ndarray) -> float:
        state = ActiveDensityState(self.model.scf_space, params)
        direction_state = ActiveDensityState(self.model.scf_space, direction)
        return self.correction_expectation(
            self.model._active_density_from_state(direction_state),
            self.model._mean_field_from_state(state),
        )

    def evaluate_state(
        self, input_state: ActiveDensityState, mu_guess: float
    ) -> tuple[DensityResult, ActiveDensityState, EnergyEvaluation]:
        correction = self.model._mean_field_from_state(input_state)
        density = self.evaluate_mean_field(correction, mu_guess)
        output_state = self.state_from_density(density.entries)
        one_body = density.band_energy - self.correction_expectation(
            self.model._active_density_from_state(output_state), correction
        )
        internal_energy = one_body + self.interaction_energy(output_state.values)
        energy = EnergyEvaluation(
            linear_free_energy=one_body - self.model.kT * density.entropy,
            internal_energy=internal_energy,
            free_energy=internal_energy - self.model.kT * density.entropy,
        )
        return density, output_state, energy
