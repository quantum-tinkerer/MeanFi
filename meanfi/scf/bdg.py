from __future__ import annotations

import numpy as np

from meanfi.density.density import solve_bdg_density_fixed_filling
from meanfi.model import Model
from meanfi.results import DensityMatrixResult
from meanfi.scf.engine import SCFProblem, SolverRuntime, warn_on_projection
from meanfi.space import MeanFieldDensitySpace
from meanfi.tb.ops import _tb_type


def build_bdg_scf_problem(model: Model, runtime: SolverRuntime) -> SCFProblem:
    """Build the superconducting map consumed by the generic SCF engine."""

    space = MeanFieldDensitySpace.bdg(model)

    def project_guess(guess: _tb_type) -> _tb_type:
        projected = space.project_guess(guess)
        warn_on_projection(guess, projected, label="BdG SCF guess")
        return projected

    def evaluate_meanfield(
        meanfield_guess: _tb_type,
        *,
        mu_guess: float,
    ) -> DensityMatrixResult:
        return solve_bdg_density_fixed_filling(
            model,
            meanfield_guess,
            keys=space.density_keys,
            integration=runtime.integration,
            filling_tol=runtime.filling_tol,
            mu_tol=runtime.mu_tol,
            max_charge_evaluations=runtime.max_charge_evaluations,
            mu_guess=mu_guess,
            density_selection=space.density_selection_for(meanfield_guess),
        )

    def evaluate_projected_guess(projected_guess: _tb_type) -> DensityMatrixResult:
        return evaluate_meanfield(projected_guess, mu_guess=0.0)

    def params_from_density_result(density_result: DensityMatrixResult) -> np.ndarray:
        return space.params_from_density(density_result.density_matrix)

    def density_result_from_params(
        params: np.ndarray, mu_guess: float
    ) -> DensityMatrixResult:
        return evaluate_meanfield(
            space.meanfield_from_density(space.density_from_params(params)),
            mu_guess=mu_guess,
        )

    def finalize_meanfield(density_result: DensityMatrixResult) -> _tb_type:
        return space.meanfield_from_density(density_result.density_matrix)

    return SCFProblem(
        runtime=runtime,
        project_guess=project_guess,
        evaluate_projected_guess=evaluate_projected_guess,
        params_from_density_result=params_from_density_result,
        density_result_from_params=density_result_from_params,
        finalize_meanfield=finalize_meanfield,
    )
