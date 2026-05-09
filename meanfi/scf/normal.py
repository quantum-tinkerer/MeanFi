from __future__ import annotations

import numpy as np

from meanfi.density.density import solve_density_matrix_fixed_filling
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.results import DensityMatrixResult
from meanfi.scf.engine import (
    SCFProblem,
    SolverRuntime,
    warn_on_projection,
)
from meanfi.space import MeanFieldDensitySpace
from meanfi.tb.ops import _tb_type


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
    density_selection=None,
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
        density_selection=density_selection,
    )


def build_normal_scf_problem(model: Model, runtime: SolverRuntime) -> SCFProblem:
    """Build the normal-state map consumed by the generic SCF engine."""

    space = MeanFieldDensitySpace.normal(model)
    keys = space.keys

    def project_guess(guess: _tb_type) -> _tb_type:
        projected = space.project_guess(guess)
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
        kwargs["density_selection"] = space.density_selection_for(hamiltonian)
        return _density_update_for_normal_hamiltonian(model, hamiltonian, **kwargs)

    def evaluate_projected_guess(projected_guess: _tb_type) -> DensityMatrixResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_meanfield(projected_guess),
            mu_guess=0.0,
        )

    def params_from_density_result(density_result: DensityMatrixResult) -> np.ndarray:
        return space.params_from_density(density_result.density_matrix)

    def density_result_from_params(
        params: np.ndarray, mu_guess: float
    ) -> DensityMatrixResult:
        return evaluate_hamiltonian(
            model.hamiltonian_from_rho(space.density_from_params(params)),
            mu_guess=mu_guess,
        )

    def finalize_meanfield(density_result: DensityMatrixResult) -> _tb_type:
        return space.meanfield_from_density(
            density_result.density_matrix,
            mu=density_result.mu,
        )

    return SCFProblem(
        runtime=runtime,
        project_guess=project_guess,
        evaluate_projected_guess=evaluate_projected_guess,
        params_from_density_result=params_from_density_result,
        density_result_from_params=density_result_from_params,
        finalize_meanfield=finalize_meanfield,
    )
