from __future__ import annotations

import numpy as np

from meanfi.density.density import solve_density_matrix_fixed_filling
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.meanfield import meanfield
from meanfi.model import Model
from meanfi.results import DensityMatrixResult
from meanfi.scf.engine import (
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
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
    density_coordinates=None,
) -> DensityMatrixResult:
    return solve_density_matrix_fixed_filling(
        hamiltonian,
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_guess=mu_guess,
        density_coordinates=density_coordinates,
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
    ) -> DensityMatrixResult:
        kwargs = dict(
            keys=keys,
            integration=runtime.integration,
            filling_tol=runtime.filling_tol,
            mu_tol=runtime.mu_tol,
            max_charge_evaluations=runtime.max_charge_evaluations,
            mu_guess=mu_guess,
        )
        kwargs["density_coordinates"] = space.required_density_coordinates_for(hamiltonian)
        return _density_update_for_normal_hamiltonian(model, hamiltonian, **kwargs)

    def evaluate_projected_guess(projected_guess: _tb_type) -> DensityMatrixResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_meanfield(projected_guess),
            mu_guess=0.0,
        )

    def density_result_from_params(
        params: np.ndarray, mu_guess: float
    ) -> DensityMatrixResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_rho(space.meanfield_input_from_params(params)),
            mu_guess=mu_guess,
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
    )


def _meanfield_from_active_density(
    active_density: _tb_type,
    *,
    model: Model,
    interaction_keys: list[tuple[int, ...]],
    onsite: tuple[int, ...],
    mu: float,
) -> _tb_type:
    zero = np.zeros((model._ndof, model._ndof), dtype=complex)
    density_reduced = {
        key: active_density.get(key, zero) for key in interaction_keys
    }
    result = dict(meanfield(density_reduced, model.h_int))
    result[onsite] = result.get(
        onsite,
        np.zeros((model._ndof, model._ndof), dtype=complex),
    ) - float(mu) * np.eye(model._ndof)
    return result
