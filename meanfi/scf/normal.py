from __future__ import annotations

import numpy as np

from meanfi.density.density import solve_density_matrix_fixed_filling
from meanfi.errors import ErrorTolerances
from meanfi.density.integrate.methods import AdaptiveSimplex, IntegrationMethod
from meanfi.meanfield import meanfield, reference_subtracted_density
from meanfi.model import Model
from meanfi.observables import expectation_value
from meanfi.results import DensityMatrixResult
from meanfi.scf.engine import (
    EnergyEvaluation,
    SCFProblem,
    SolverRuntime,
    warn_on_projection,
)
from meanfi.tb.ops import _tb_type
from meanfi.tb.storage import match_tb_storage, prefers_sparse_storage


def _density_update_for_normal_hamiltonian(
    model: Model,
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    tolerances: ErrorTolerances,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
    density_coordinates=None,
    include_band_energy: bool = False,
) -> DensityMatrixResult:
    return solve_density_matrix_fixed_filling(
        hamiltonian,
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_guess=mu_guess,
        density_coordinates=density_coordinates,
        include_band_energy=include_band_energy,
    )


def build_normal_scf_problem(model: Model, runtime: SolverRuntime) -> SCFProblem:
    """Build the normal-state map consumed by the generic SCF engine."""

    space = model.scf_space
    keys = space.interaction_keys

    def project_guess(guess: _tb_type) -> _tb_type:
        projected = space.project_meanfield_input(guess)
        projected = match_tb_storage(
            projected,
            like_sparse=prefers_sparse_storage(
                getattr(model, "h_0", None),
                model.h_int,
                guess,
            ),
        )
        warn_on_projection(guess, projected, label="Normal SCF guess")
        return projected

    def evaluate_hamiltonian(
        hamiltonian: _tb_type,
        *,
        mu_guess: float,
        include_band_energy: bool = False,
    ) -> DensityMatrixResult:
        kwargs = dict(
            keys=keys,
            integration=runtime.integration,
            tolerances=runtime.tolerances,
            mu_tol=runtime.mu_tol,
            max_charge_evaluations=runtime.max_charge_evaluations,
            mu_guess=mu_guess,
        )
        if include_band_energy:
            kwargs["include_band_energy"] = True
        if isinstance(runtime.integration, AdaptiveSimplex):
            kwargs["density_coordinates"] = space.required_coordinates
        else:
            kwargs["density_coordinates"] = space.required_density_coordinates_for(
                hamiltonian
            )
        return _density_update_for_normal_hamiltonian(model, hamiltonian, **kwargs)

    def evaluate_projected_guess(projected_guess: _tb_type) -> DensityMatrixResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_meanfield(projected_guess),
            mu_guess=0.0,
        )

    def density_result_from_params(
        params: np.ndarray,
        mu_guess: float,
        *,
        include_band_energy: bool = False,
    ) -> DensityMatrixResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_rho(space.meanfield_input_from_params(params)),
            mu_guess=mu_guess,
            include_band_energy=include_band_energy,
        )

    def density_difference(params: np.ndarray) -> _tb_type:
        return reference_subtracted_density(
            space.meanfield_input_from_params(params),
            getattr(model, "reference_density_matrix", None),
            interaction_keys=space.interaction_keys,
            onsite=space.onsite,
            ndof=model._ndof,
        )

    def interaction_energy(params: np.ndarray) -> float:
        difference = density_difference(params)
        correction = meanfield(difference, model.h_int)
        return float(0.5 * np.real(expectation_value(difference, correction)))

    def interaction_gradient(
        params: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        correction = meanfield(density_difference(params), model.h_int)
        direction_density = space.meanfield_input_from_params(direction)
        return float(np.real(expectation_value(direction_density, correction)))

    def energy_from_result(
        input_params: np.ndarray,
        density_result: DensityMatrixResult,
    ) -> EnergyEvaluation:
        if density_result.band_energy is None:
            raise RuntimeError("EnergyDIIS requires an occupied band-energy result")
        output_params = np.asarray(
            space.params_from_meanfield_input(density_result.density_matrix),
            dtype=float,
        )
        output_density = space.meanfield_input_from_params(output_params)
        input_correction = meanfield(density_difference(input_params), model.h_int)
        one_body = float(
            density_result.band_energy
            - np.real(expectation_value(output_density, input_correction))
        )
        energy = one_body + interaction_energy(output_params)
        return EnergyEvaluation(
            params=output_params,
            one_body_energy=one_body,
            energy=energy,
        )

    def finalize_meanfield(density_result: DensityMatrixResult) -> _tb_type:
        return _meanfield_from_active_density(
            space.project_meanfield_input(density_result.density_matrix),
            model=model,
            interaction_keys=space.interaction_keys,
            onsite=space.onsite,
            mu=density_result.mu,
        )

    return SCFProblem(
        runtime=runtime,
        project_guess=project_guess,
        evaluate_projected_guess=evaluate_projected_guess,
        compress_density=space.params_from_meanfield_input,
        density_result_from_params=density_result_from_params,
        finalize_meanfield=finalize_meanfield,
        energy_from_result=energy_from_result,
        interaction_energy=interaction_energy,
        interaction_gradient=interaction_gradient,
    )


def _meanfield_from_active_density(
    active_density: _tb_type,
    *,
    model: Model,
    interaction_keys: list[tuple[int, ...]],
    onsite: tuple[int, ...],
    mu: float,
) -> _tb_type:
    density_reduced = reference_subtracted_density(
        active_density,
        getattr(model, "reference_density_matrix", None),
        interaction_keys=interaction_keys,
        onsite=onsite,
        ndof=model._ndof,
    )
    result = dict(meanfield(density_reduced, model.h_int))
    result[onsite] = result.get(
        onsite,
        np.zeros((model._ndof, model._ndof), dtype=complex),
    ) - float(mu) * np.eye(model._ndof)
    return result
