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
    SCFIterationInfo,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
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
            self.scf_space = FakeSpace(self)

        def hamiltonian_from_meanfield(self, mf):
            return mf

        def hamiltonian_from_rho(self, rho):
            return rho

    class FakeSpace:
        interaction_keys = [(0,)]
        onsite = (0,)

        def __init__(self, model):
            self.model = model

        def project_meanfield_input(self, guess):
            return guess

        def required_density_coordinates_for(self, hamiltonian):
            del hamiltonian
            return None

        def params_from_meanfield_input(self, rho):
            return np.asarray(rho[(0,)], dtype=float)

        def meanfield_input_from_params(self, params):
            return {(0,): np.asarray(params, dtype=float)}

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
        density_coordinates,
    ):
        del (
            keys,
            integration,
            filling_tol,
            mu_tol,
            max_charge_evaluations,
            mu_guess,
            density_coordinates,
        )
        return fake_result(hamiltonian, model.step)

    monkeypatch.setattr(
        normal_scf,
        "_density_update_for_normal_hamiltonian",
        fake_density_for_hamiltonian,
    )
    monkeypatch.setattr(
        normal_scf,
        "_meanfield_from_active_density",
        lambda *args, **kwargs: {},
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
    assert np.isclose(info_short.history[-1].line_search_norm, np.hypot(0.1, 0.02))
    assert np.isclose(info_long.residual_norm, 0.1)
    assert info_short.total_unique_evals == info_long.total_unique_evals == 0


def test_rejected_line_search_trial_does_not_update_scf_state(monkeypatch):
    import meanfi.scf.engine as scf_engine
    from meanfi.scf.info import SCFRunState

    mu_guesses = []

    def fake_result(params, mu_guess):
        value = float(np.ravel(params)[0])
        mu_guesses.append((value, float(mu_guess)))
        return DensityMatrixResult(
            density_matrix={(0,): np.array([value + 0.25])},
            density_matrix_error=None,
            mu=10.0 + value,
            filling=1.0,
            target_filling=1.0,
            filling_residual=0.0,
            integration=AdaptiveQuadrature(),
            info=AdaptiveQuadratureInfo(
                n_kernel_evals=0,
                unique_evals=1,
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

    def fake_solve_fixed_point(residual_fn, x0, *, scf, scf_tol, on_iteration):
        del scf, scf_tol
        initial = np.asarray(x0, dtype=float)
        residual = residual_fn(initial)
        on_iteration(None, float(np.max(np.abs(residual))), initial, residual)

        residual_fn(np.array([99.0]))

        accepted = np.array([1.0])
        residual = residual_fn(accepted)
        on_iteration(1, float(np.max(np.abs(residual))), accepted, residual)
        return accepted

    monkeypatch.setattr(scf_engine, "solve_fixed_point", fake_solve_fixed_point)

    run = scf_engine.iterate_density_fixed_point(
        np.array([0.0]),
        density_result_from_params=fake_result,
        compress_density=lambda density: density[(0,)],
        scf=AndersonMixing(),
        scf_tol=1e-8,
        state=SCFRunState(mu=2.0),
    )

    assert mu_guesses == [(0.0, 2.0), (99.0, 10.0), (1.0, 10.0), (1.0, 11.0)]
    assert run.state.mu == 11.0
    assert [item.mu for item in run.state.history] == [10.0, 11.0, 11.0]


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


def test_solver_info_exposes_scf_iteration_history():
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

    history = result.info.history
    assert history
    assert all(isinstance(item, SCFIterationInfo) for item in history)
    assert np.isclose(history[-1].residual_norm, result.info.residual_norm)
    assert [item.step for item in history] == list(range(1, len(history) + 1))
    assert all(item.integration_evals >= 0 for item in history)
    assert all(item.line_search_norm >= item.residual_norm for item in history)
    assert history[-1].charge_error is not None
    assert [item.cumulative_integration_evals for item in history] == list(
        np.cumsum([item.integration_evals for item in history])
    )


def test_solver_verbose_prints_scf_progress(capsys):
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
        verbose=True,
    )

    output = capsys.readouterr().out
    assert result.info.history
    assert "scf step=1" in output
    assert "residual=" in output
    assert "line_search_norm=" in output
    assert "integration_evals=" in output
    assert "mu=" in output
    assert "charge_error=" in output
