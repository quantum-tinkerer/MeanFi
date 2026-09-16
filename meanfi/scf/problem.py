"""Shared normal/BdG SCF evaluation and its physical energy functional."""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np

from meanfi.density.density import evaluate_density
from meanfi.density.problem import DensityProblem
from meanfi.model import Model
from meanfi.meanfield import interaction_energy
from meanfi.observables import _internal_energy_from_band
from meanfi.results import _DensityEntries, DensityResult
from meanfi.space.state import ActiveDensityState
from meanfi.tb.ops import _tb_type, add_tb
from meanfi.tb.storage import tb_entries_changed


@dataclass(frozen=True, kw_only=True)
class SCFEvaluation:
    """One density evaluation and its input correction; initial input density is unknown."""

    density: DensityResult
    output_state: ActiveDensityState
    internal_energy: float
    mean_field: _tb_type
    input_state: ActiveDensityState | None = None

    @property
    def residual(self) -> np.ndarray | None:
        if self.input_state is None:
            return None
        return self.output_state.values - self.input_state.values

    @property
    def residual_norm(self) -> float | None:
        residual = self.residual
        return (
            None if residual is None else self.output_state.space.density_norm(residual)
        )


@dataclass(frozen=True)
class SCFProblem:
    model: Model
    density_problem: DensityProblem
    max_charge_evaluations: int | None = None

    def project_guess(self, guess: _tb_type) -> _tb_type:
        model = self.model
        model._validate_mean_field(guess)
        projected = model._project_mean_field(guess)
        if tb_entries_changed(guess, projected):
            warnings.warn(
                "SCF guess contains values outside the active SCF density selection; "
                "those values were projected away before the first iteration",
                UserWarning,
                stacklevel=3,
            )
        return projected

    def evaluate_mean_field(
        self,
        mean_field: _tb_type,
        mu_guess: float,
        *,
        mu: float | None = None,
        compute_entropy: bool = False,
    ) -> DensityResult:
        return evaluate_density(
            replace(
                self.density_problem,
                hamiltonian=add_tb(self.model._hamiltonian, mean_field),
            ),
            filling=self.model.filling if mu is None else None,
            mu=mu,
            compute_entropy=compute_entropy,
            max_charge_evaluations=self.max_charge_evaluations,
            mu_guess=mu_guess,
        )

    def state_from_density(self, density: _DensityEntries) -> ActiveDensityState:
        space = self.model._space
        if density.coordinates is not space.required_coordinates:
            raise ValueError("density slice does not match the SCF space")
        return ActiveDensityState(
            space, space.params_from_required_entries(density.values)
        )

    def interaction_curvature(self, difference: np.ndarray) -> float:
        """Quadratic energy of a direction; the reference cancels in EDIIS differences."""
        return interaction_energy(
            self.model._space.density_from_params(difference),
            self.model.h_int,
            electron_ndof=self.model._electron_ndof,
        )

    def evaluate_state(
        self, input_state: ActiveDensityState, mu_guess: float
    ) -> SCFEvaluation:
        model = self.model
        correction = model._mean_field_from_state(input_state)
        density = self.evaluate_mean_field(correction, mu_guess)
        output_state = self.state_from_density(density.entries)
        energy = _internal_energy_from_band(
            model, output_state, density.band_energy, correction
        )
        return SCFEvaluation(
            density=density,
            output_state=output_state,
            input_state=input_state,
            internal_energy=energy,
            mean_field=correction,
        )
