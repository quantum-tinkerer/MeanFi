# ruff: noqa: F401
import importlib
import inspect
from types import SimpleNamespace

import meanfi
import numpy as np
import pytest
import scipy.sparse as sp

from meanfi import (
    AdaptiveQuadrature,
    AdaptiveQuadratureInfo,
    AdaptiveSimplex,
    AndersonMixing,
    DensityMatrixResult,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    guess_tb,
    solver,
)
from meanfi.density.filling import mu_bracket, solve_mu
from meanfi.density.integrate.quadrature.normal import resolve_normal_matrix_function
from meanfi.density.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.density.integrate.uniform import resolve_uniform_grid_matrix_function
from meanfi.scf.engine import NoConvergence
from meanfi.tb.ops import matrix_bound
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def test_solver_raises_no_convergence_when_scf_budget_is_exhausted():
    model = Model(
        spinful_chain(),
        {(0,): np.eye(2)},
        filling=1.0,
        kT=0.1,
    )

    with pytest.raises(NoConvergence) as exc_info:
        solver(
            model,
            {(0,): 0.2 * np.eye(2)},
            integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
            scf=LinearMixing(max_iterations=1, alpha=0.1),
            scf_tol=1e-30,
        )

    assert exc_info.value.last_iterate.size > 0


def test_solver_info_residual_norm_uses_max_norm_and_is_not_extensive(monkeypatch):
    import meanfi.scf.scf as scf_pipeline
    import meanfi.scf.normal as normal_scf

    def fake_result(hamiltonian, step):
        params = np.asarray(hamiltonian[(0,)], dtype=float)
        return DensityMatrixResult(
            density_matrix={(0,): params + step},
            density_matrix_error=None,
            mu=0.0,
            filling=1.0,
            target_filling=1.0,
            filling_residual=0.0,
            integration=AdaptiveQuadrature(),
            info=AdaptiveQuadratureInfo(
                n_kernel_evals=0,
                unique_evals=0,
                n_evaluator_evals=0,
                n_cached_nodes=0,
                n_leaves=0,
                n_leaf_nodes=0,
                refinements=0,
                error_estimate_available=True,
                charge_integration_calls=0,
                density_integration_calls=1,
            ),
        )

    class FakeModel:
        def __init__(self, step):
            self.step = np.asarray(step, dtype=float)
            self.h_int = {(0,): np.zeros((1, 1))}
            self._ndof = 1
            self._local_key = (0,)
            self.filling = 1.0
            self.kT = 0.2

        def hamiltonian_from_meanfield(self, mf):
            return mf

        def hamiltonian_from_rho(self, rho):
            return rho

    class FakeSpace:
        keys = [(0,)]

        def __init__(self, model):
            self.model = model

        def project_guess(self, guess):
            return guess

        def density_selection_for(self, hamiltonian):
            del hamiltonian
            return None

        def params_from_density(self, rho):
            return np.asarray(rho[(0,)], dtype=float)

        def density_from_params(self, params):
            return {(0,): np.asarray(params, dtype=float)}

        def meanfield_from_density(self, rho, *, mu=0.0):
            del rho, mu
            return {}

    def fake_density_for_hamiltonian(
        model,
        hamiltonian,
        *,
        keys,
        integration,
        filling_tol,
        mu_tol,
        max_charge_evaluations,
        mu_guess,
        density_selection,
    ):
        del (
            keys,
            integration,
            filling_tol,
            mu_tol,
            max_charge_evaluations,
            mu_guess,
            density_selection,
        )
        return fake_result(hamiltonian, model.step)

    monkeypatch.setattr(
        normal_scf.MeanFieldDensitySpace,
        "normal",
        lambda model: FakeSpace(model),
    )
    monkeypatch.setattr(
        normal_scf,
        "_density_update_for_normal_hamiltonian",
        fake_density_for_hamiltonian,
    )

    info_short = scf_pipeline.solver(
        FakeModel([0.1, -0.02]),
        {(0,): np.zeros(2)},
        integration=AdaptiveQuadrature(),
        scf=LinearMixing(max_iterations=1),
        scf_tol=0.2,
    ).info
    info_long = scf_pipeline.solver(
        FakeModel([0.1, -0.02, 0.1, -0.02, 0.1, -0.02]),
        {(0,): np.zeros(6)},
        integration=AdaptiveQuadrature(),
        scf=LinearMixing(max_iterations=1),
        scf_tol=0.2,
    ).info

    assert np.isclose(info_short.residual_norm, 0.1)
    assert np.isclose(info_long.residual_norm, 0.1)
    assert info_short.total_unique_evals == info_long.total_unique_evals == 0


def test_solver_info_exposes_total_unique_evals():
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2))},
        filling=1.0,
        kT=0.1,
    )
    result = solver(
        model,
        {(0,): np.zeros((2, 2))},
        integration=AdaptiveQuadrature(density_matrix_tol=1e-5),
        scf=LinearMixing(max_iterations=3),
        scf_tol=1e-5,
    )

    assert (
        result.info.total_unique_evals >= result.density_matrix_result.info.unique_evals
    )
    assert result.info.total_unique_evals > 0
