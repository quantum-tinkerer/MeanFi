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
    AdaptiveSimplexInfo,
    AndersonMixing,
    DensityMatrixResult,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
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
from meanfi.scf.engine import SolverRuntime
from meanfi.scf.normal import build_normal_scf_problem
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import matrix_bound
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def test_normal_solver_warns_when_guess_is_projected_to_structural_selection():
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.2,
    )

    with pytest.warns(UserWarning, match="projected away"):
        result = solver(
            model,
            {(0,): np.array([[0.0, 0.3], [0.3, 0.0]], dtype=complex)},
            integration=AdaptiveQuadrature(density_matrix_tol=1e-2),
            scf=LinearMixing(max_iterations=1),
            scf_tol=1e-8,
        )

    assert result.info.iterations >= 1


def test_density_matrix_requires_local_key_for_zero_dimensional_inputs():
    with pytest.raises(ValueError, match="local key"):
        density_matrix_at_mu(
            {(): np.diag([-1.0, 1.0]), (1,): np.ones((2, 2))},
            mu=0.0,
            kT=0.1,
            keys=[()],
            integration=AdaptiveQuadrature(),
        )


@requires_ext
@pytest.mark.parametrize("mode", ("at_mu", "fixed_filling"))
def test_root_mesh_only_zero_temperature_mode_reports_no_error_estimate(mode):
    tb = spinful_chain()
    integration = AdaptiveSimplex(density_matrix_tol=1e-6, max_refinements=0)
    keys = [(0,), (1,), (-1,)]

    if mode == "at_mu":
        result = density_matrix_at_mu(
            tb,
            mu=0.2,
            kT=0.0,
            keys=keys,
            integration=integration,
        )
        assert result.info.refinements == 0
        assert result.info.error_estimate_available is False
        assert result.density_matrix_error is None
        assert np.allclose(
            result.density_matrix[(-1,)],
            result.density_matrix[(1,)].conj().T,
            atol=1e-8,
        )
        return

    result = density_matrix(
        tb,
        filling=1.0,
        kT=0.0,
        keys=keys,
        integration=integration,
    )
    assert np.isfinite(result.mu)
    assert result.info.refinements == 0
    assert result.info.error_estimate_available is False
    assert result.density_matrix_error is None
    assert np.allclose(
        result.density_matrix[(-1,)],
        result.density_matrix[(1,)].conj().T,
        atol=1e-8,
    )


def test_positive_temperature_density_matrix_does_not_use_zero_temperature_backend(
    monkeypatch,
):
    import meanfi.density.integrate.normal as integration

    def fail(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError(
            "AdaptiveQuadrature should not call the zero-temperature backend"
        )

    monkeypatch.setattr(integration, "density_matrix_zero_temp", fail)
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.1,
        keys=[(0,)],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-4),
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - 1.0) <= 2e-4
    assert np.allclose(
        result.density_matrix[(0,)],
        result.density_matrix[(0,)].conj().T,
        atol=1e-8,
    )


def test_zero_temperature_density_matrix_dispatches_to_zero_temperature_backend(
    monkeypatch,
):
    import meanfi.density.integrate.normal as integration

    called = {}

    def fake_density_matrix_zero_temp(*args, **kwargs):
        called["kwargs"] = kwargs
        return (
            {(0,): np.array([[1.0]])},
            {(0,): np.array([[0.0]])},
            0.0,
            SimpleNamespace(
                mu=0.0,
                charge=1.0,
                n_kernel_evals=1,
                unique_evals=1,
                n_evaluator_evals=1,
                n_cached_nodes=1,
                n_leaves=1,
                n_leaf_nodes=1,
                subdivisions=0,
                error_estimate_available=True,
                charge_integration_calls=1,
                density_integration_calls=1,
            ),
        )

    monkeypatch.setattr(
        integration, "density_matrix_zero_temp", fake_density_matrix_zero_temp
    )
    result = density_matrix(
        {(0,): np.zeros((1, 1)), (1,): np.zeros((1, 1)), (-1,): np.zeros((1, 1))},
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        integration=AdaptiveSimplex(density_matrix_tol=1e-4),
    )

    assert called["kwargs"]["density_atol"] == 1e-4
    assert np.allclose(result.density_matrix[(0,)], np.array([[1.0]]))
    assert np.allclose(result.density_matrix_error[(0,)], np.array([[0.0]]))
    assert result.mu == 0.0
    assert result.filling == 1.0


def test_adaptive_simplex_scf_passes_required_coordinates_for_dense_hamiltonian(
    monkeypatch,
):
    import meanfi.scf.normal as normal_scf

    required = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,)],
        pairs_by_key={(0,): (np.array([0]), np.array([0]))},
        allow_empty=False,
    )
    captured = {}

    class FakeSpace:
        interaction_keys = [(0,)]
        onsite = (0,)
        required_coordinates = required

        def meanfield_input_from_params(self, params):
            del params
            return {(0,): np.zeros((2, 2), dtype=complex)}

        def required_density_coordinates_for(self, hamiltonian):
            del hamiltonian
            raise AssertionError("AdaptiveSimplex should use required_coordinates directly")

        def project_meanfield_input(self, tb):
            return tb

        def params_from_meanfield_input(self, rho):
            return np.array([rho[(0,)][0, 0].real])

    class FakeModel:
        filling = 1.0
        kT = 0.0
        h_int = {(0,): np.zeros((2, 2), dtype=complex)}
        h_0 = {(0,): np.zeros((2, 2), dtype=complex)}
        _ndof = 2
        scf_space = FakeSpace()

        def hamiltonian_from_rho(self, rho):
            del rho
            return {(0,): np.zeros((2, 2), dtype=complex)}

    def fake_density_update(*args, **kwargs):
        del args
        captured["density_coordinates"] = kwargs["density_coordinates"]
        return DensityMatrixResult(
            density_matrix={(0,): np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)},
            density_matrix_error=None,
            mu=0.0,
            filling=1.0,
            target_filling=1.0,
            filling_residual=0.0,
            integration=AdaptiveSimplex(),
            info=AdaptiveSimplexInfo(
                n_kernel_evals=0,
                unique_evals=0,
                n_evaluator_evals=0,
                n_cached_nodes=0,
                n_leaves=0,
                n_leaf_nodes=0,
                refinements=0,
                error_estimate_available=True,
                charge_evaluations=0,
                charge_integration_calls=0,
                density_integration_calls=1,
            ),
        )

    monkeypatch.setattr(
        normal_scf,
        "_density_update_for_normal_hamiltonian",
        fake_density_update,
    )
    problem = build_normal_scf_problem(
        FakeModel(),
        SolverRuntime(
            integration=AdaptiveSimplex(),
            filling_tol=None,
            mu_tol=1e-10,
            max_charge_evaluations=None,
        ),
    )

    problem.density_result_from_params(np.zeros(1), 0.0)

    assert captured["density_coordinates"] is required


def test_adaptive_simplex_wrapper_passes_generic_density_components(monkeypatch):
    import meanfi.density.integrate.simplex as simplex_integration

    required = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,), (1,)],
        pairs_by_key={
            (0,): (np.array([0]), np.array([1])),
            (1,): (np.array([1]), np.array([0])),
        },
        allow_empty=False,
    )
    captured = {}

    def fake_density_matrix_zero_temp(*args, **kwargs):
        del args
        captured.update(kwargs)
        return (
            {(0,): np.zeros((2, 2), dtype=complex), (1,): np.zeros((2, 2), dtype=complex)},
            {(0,): np.zeros((2, 2), dtype=float), (1,): np.zeros((2, 2), dtype=float)},
            0.0,
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        simplex_integration,
        "_density_matrix_zero_temp",
        fake_density_matrix_zero_temp,
    )

    simplex_integration.density_matrix_zero_temp(
        {
            (0,): np.zeros((2, 2), dtype=complex),
            (1,): np.zeros((2, 2), dtype=complex),
        },
        filling=1.0,
        keys=[(0,), (1,)],
        density_coordinates=required,
        charge_tol=1e-3,
        density_atol=1e-3,
        density_rtol=0.0,
        mu_guess=0.0,
        mu_xtol=1e-10,
        max_charge_evaluations=None,
    )

    assert captured["density_components"] == [(0, 1, (0,)), (1, 0, (1,))]
    assert "density_coordinates" not in captured


def test_zero_temperature_runtime_error_when_extension_missing(monkeypatch):
    import adaptivesimplex.backend as simplex_backend

    monkeypatch.setattr(simplex_backend, "NATIVE_AVAILABLE", False)
    monkeypatch.setattr(simplex_backend, "Geometry", None)

    with pytest.raises(
        RuntimeError,
        match="requires the compiled adaptivesimplex._native extension",
    ):
        density_matrix(
            spinful_chain(),
            filling=1.0,
            kT=0.0,
            keys=[(0,)],
            integration=AdaptiveSimplex(density_matrix_tol=1e-4),
        )


@requires_ext
def test_zero_temperature_backend_supports_higher_dimensions():
    result = density_matrix_at_mu(
        {(0, 0, 0, 0): np.diag([-1.0, 1.0])},
        mu=0.0,
        kT=0.0,
        keys=[(0, 0, 0, 0)],
        integration=AdaptiveSimplex(density_matrix_tol=1e-12, max_refinements=10),
    )

    assert np.allclose(
        result.density_matrix[(0, 0, 0, 0)],
        np.diag([1.0, 0.0]),
        atol=1e-12,
    )
    assert result.info.n_leaves > 0
