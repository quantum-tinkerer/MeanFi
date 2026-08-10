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
    AdaptiveSimplex,
    AndersonMixing,
    DirectDiagonalization,
    LinearMixing,
    default_solver_tolerances,
    Model,
    RationalFOE,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
)
from meanfi.density.filling import mu_bracket, solve_mu
from meanfi.density.internal import DensityEvaluation, DensitySlice
from meanfi.errors import ErrorValues
from meanfi.density.integrate.quadrature.normal import resolve_normal_matrix_function
from meanfi.density.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.density.integrate.uniform import resolve_uniform_grid_matrix_function
from meanfi.scf.engine import NoConvergence
from meanfi.scf.engine import SolverRuntime
from meanfi.scf.normal import build_normal_scf_problem
from meanfi.space.state import ActiveDensityState
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

    assert result.history


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
def test_zero_temperature_backend_raises_when_refinement_cap_prevents_convergence(mode):
    tb = spinful_chain()
    integration = AdaptiveSimplex(density_matrix_tol=1e-6, max_refinements=0)
    keys = [(0,), (1,), (-1,)]

    with pytest.raises(RuntimeError, match="Adaptive simplex loop did not converge"):
        if mode == "at_mu":
            density_matrix_at_mu(
                tb,
                mu=0.2,
                kT=0.0,
                keys=keys,
                integration=integration,
            )
            return

        density_matrix(
            tb,
            filling=1.0,
            kT=0.0,
            keys=keys,
            integration=integration,
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
                num_threads=3,
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
        integration=AdaptiveSimplex(density_matrix_tol=1e-4, num_threads=3),
        filling_tol=2e-3,
    )

    assert called["kwargs"]["density_atol"] == 1e-4
    assert called["kwargs"]["charge_tol"] == 1e-5
    assert called["kwargs"]["filling_tol"] == 2e-3
    assert called["kwargs"]["num_threads"] == 3
    assert np.allclose(result.density_matrix[(0,)], np.array([[1.0]]))
    assert result.errors.density_matrix_integration == 0.0
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
        density_keys = [(0,)]
        onsite = (0,)
        required_coordinates = required

        def meanfield_input_from_params(self, params):
            del params
            return {(0,): np.zeros((2, 2), dtype=complex)}

        def required_density_coordinates_for(self, hamiltonian):
            del hamiltonian
            raise AssertionError(
                "AdaptiveSimplex should use required_coordinates directly"
            )

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
        return DensityEvaluation(
            density=DensitySlice(
                required,
                np.array([1.0], dtype=complex),
                np.array([0.0]),
            ),
            mu=0.0,
            filling=1.0,
            errors=ErrorValues(
                density_matrix_integration=0.0,
                filling_residual=0.0,
                charge_integration=0.0,
            ),
            integration=AdaptiveSimplex(),
            statistics=SimpleNamespace(),
            band_energy=0.0,
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
            tolerances=default_solver_tolerances(1e-3),
            mu_tol=1e-10,
            max_charge_evaluations=None,
        ),
    )

    problem.evaluate_state(ActiveDensityState(problem.state_space, np.zeros(1)), 0.0)

    assert captured["density_coordinates"] is required


def test_adaptive_simplex_maps_selected_density_results_to_tb():
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
    native_result = SimpleNamespace(
        values=np.array([2.0, 7.0], dtype=complex),
        stopping_error=2e-4,
    )

    density, error = simplex_integration._density_result_to_tb(
        native_result,
        required,
    )

    assert np.array_equal(density[(0,)], np.array([[0.0, 2.0], [0.0, 0.0]]))
    assert np.array_equal(density[(1,)], np.array([[0.0, 0.0], [7.0, 0.0]]))
    assert np.array_equal(error[(0,)], np.array([[0.0, 2e-4], [0.0, 0.0]]))
    assert np.array_equal(error[(1,)], np.array([[0.0, 0.0], [2e-4, 0.0]]))


def test_adaptive_simplex_empty_density_selection_reports_no_density_call(monkeypatch):
    import meanfi.density.integrate.simplex as simplex_integration

    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,)],
        pairs_by_key={
            (0,): (
                np.array([], dtype=int),
                np.array([], dtype=int),
            )
        },
        allow_empty=True,
    )
    mesh = SimpleNamespace(
        cached_vertices=5,
        active_simplices=2,
        active_vertices=3,
    )
    charge_result = SimpleNamespace(
        value=1.0,
        stopping_error=1e-4,
        dcharge_dmu=0.5,
        stats=SimpleNamespace(
            evaluations=7,
            refinements=2,
            target_reached=True,
        ),
        error_stats=SimpleNamespace(hamiltonian_evaluations=3),
    )

    monkeypatch.setattr(simplex_integration, "_spectral_mesh", lambda h: mesh)
    monkeypatch.setattr(
        simplex_integration,
        "solve_mu",
        lambda **kwargs: SimpleNamespace(
            mu=0.0,
            charge=1.0,
            charge_error=0.0,
            residual=0.0,
            derivative=0.5,
            charge_evaluations=1,
        ),
    )
    monkeypatch.setattr(
        simplex_integration,
        "_integrate_charge",
        lambda *args, **kwargs: charge_result,
    )

    density, error, mu, info = simplex_integration.density_matrix_zero_temp(
        spinful_chain(),
        filling=1.0,
        keys=[(0,)],
        density_coordinates=coordinates,
        charge_tol=1e-3,
        filling_tol=1e-3,
        density_atol=1e-3,
        density_rtol=0.0,
        mu_guess=0.0,
        mu_xtol=1e-10,
        max_charge_evaluations=None,
        max_subdivisions=None,
        num_threads=None,
    )

    assert mu == 0.0
    assert np.count_nonzero(density[(0,)]) == 0
    assert np.count_nonzero(error[(0,)]) == 0
    assert info.charge_integration_calls == 1
    assert info.density_integration_calls == 0
    assert info.density_n_kernel_evals == 0


def test_adaptive_simplex_calls_fermisimplex_density_api_with_preview_depth_one():
    import meanfi.density.integrate.simplex as simplex_integration

    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,), (1,)],
        pairs_by_key={
            (0,): (np.array([0]), np.array([1])),
            (1,): (np.array([1]), np.array([0])),
        },
        allow_empty=False,
    )
    calls = []

    class Mesh:
        def integrate_density_components(self, **kwargs):
            calls.append(kwargs)
            return "density"

    result = simplex_integration._integrate_density(
        Mesh(),
        coordinates,
        mu=0.25,
        density_atol=1e-3,
        max_refinements=12,
        num_threads=None,
    )

    assert result == "density"
    assert len(calls) == 1
    components = calls[0].pop("components")
    assert np.array_equal(components, np.array([[0, 0, 1], [1, 1, 0]]))
    assert calls[0] == {
        "mu": 0.25,
        "lattice_vectors": ((0,), (1,)),
        "target_error": 1e-3,
        "max_refinements": 12,
        "preview_depth": 1,
        "min_refinement_batch_size": 1,
        "max_refinement_batch_size": 100,
    }


def test_adaptive_simplex_calls_fermisimplex_charge_apis():
    import meanfi.density.integrate.simplex as simplex_integration

    calls = []

    class Mesh:
        cached_vertices = 5

        def estimate_charge_on_current_mesh(self, **kwargs):
            calls.append(("estimate", kwargs))
            return SimpleNamespace(value=1.0, dcharge_dmu=0.5)

        def integrate_charge(self, **kwargs):
            calls.append(("integrate", kwargs))
            return "charge"

    mesh = Mesh()
    current, evaluations = simplex_integration._evaluate_charge(
        mesh,
        mu=0.25,
        num_threads=None,
    )
    adaptive = simplex_integration._integrate_charge(
        mesh,
        mu=0.5,
        charge_tol=1e-4,
        max_refinements=7,
        num_threads=None,
    )

    assert current.value == 1.0
    assert evaluations == 0
    assert adaptive == "charge"
    assert calls == [
        ("estimate", {"mu": 0.25}),
        (
            "integrate",
            {
                "mu": 0.5,
                "target_error": 1e-4,
                "max_refinements": 7,
                "error_depth": 2,
                "min_refinement_batch_size": 1,
                "max_refinement_batch_size": 100,
            },
        ),
    ]


@requires_ext
@pytest.mark.parametrize("ndim", (3, 4))
def test_zero_temperature_backend_supports_higher_dimensions(ndim):
    key = (0,) * ndim
    result = density_matrix_at_mu(
        {key: np.diag([-1.0, 1.0])},
        mu=0.0,
        kT=0.0,
        keys=[key],
        integration=AdaptiveSimplex(density_matrix_tol=1e-12, max_refinements=10),
    )

    assert np.allclose(
        result.density_matrix[key],
        np.diag([1.0, 0.0]),
        atol=1e-12,
    )
    assert result.errors.density_matrix_integration is not None
