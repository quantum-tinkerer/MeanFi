from dataclasses import replace
from meanfi import default_solver_tolerances
from meanfi.density.problem import build_density_problem
from meanfi.density.density import evaluate_density
from types import SimpleNamespace

import numpy as np
import pytest

from meanfi import (
    FermiSimplex,
    LinearMixing,
    Model,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
)
from meanfi.results import _DensityEntries, DensityResult
from meanfi.errors import ErrorValues
from meanfi.results import FermiSimplexInfo
from meanfi.scf.problem import SCFProblem
from meanfi.space.state import ActiveDensityState
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration


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
            integration=UniformGrid(density_matrix_tol=1e-2),
            scf=LinearMixing(max_iterations=1),
            scf_tol=1e-8,
        )

    assert result.history


def test_density_matrix_requires_local_key_for_zero_dimensional_inputs():
    with pytest.raises(
        ValueError, match="keys must be integer tuples of the same length"
    ):
        density_matrix_at_mu(
            {(): np.diag([-1.0, 1.0]), (1,): np.ones((2, 2))},
            mu=0.0,
            kT=0.1,
            keys=[()],
            integration=UniformGrid(),
        )


@pytest.mark.parametrize("mode", ("at_mu", "fixed_filling"))
def test_zero_temperature_backend_raises_when_refinement_cap_prevents_convergence(mode):
    tb = spinful_chain()
    integration = FermiSimplex(density_matrix_tol=1e-6, max_refinements=0)
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
    import meanfi.density.density as integration

    def fail(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError("UniformGrid should not call the zero-temperature backend")

    monkeypatch.setattr(integration, "solve_simplex", fail)
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.1,
        keys=[(0,)],
        integration=UniformGrid(density_matrix_tol=1e-4),
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - 1.0) <= 2e-4
    assert np.allclose(
        result.to_tb()[(0,)],
        result.to_tb()[(0,)].conj().T,
        atol=1e-8,
    )


def test_zero_temperature_density_matrix_dispatches_to_zero_temperature_backend(
    monkeypatch,
):
    import meanfi.density.density as integration

    called = {}

    def fake_simplex(problem, **kwargs):
        called["problem"] = problem
        return DensityResult(
            entries=_DensityEntries(
                problem.density_coordinates, np.array([1.0]), np.array([0.0])
            ),
            mu=0.0,
            filling=1.0,
            errors=ErrorValues(
                density_matrix_integration=0.0,
                charge_integration=0.0,
                filling_residual=0.0,
            ),
            statistics=FermiSimplexInfo(
                n_kernel_evals=1,
                n_cached_nodes=1,
                n_leaves=1,
                refinements=0,
                error_estimate_available=True,
                charge_integration_calls=1,
                density_integration_calls=1,
                num_threads=3,
            ),
        )

    monkeypatch.setattr(integration, "solve_simplex", fake_simplex)
    result = density_matrix(
        {(0,): np.zeros((1, 1)), (1,): np.zeros((1, 1)), (-1,): np.zeros((1, 1))},
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        integration=FermiSimplex(density_matrix_tol=1e-4, num_threads=3),
        filling_tol=2e-3,
    )

    assert called["problem"].tolerances.density_matrix_integration == 1e-4
    assert called["problem"].tolerances.charge_integration == 1e-4
    assert called["problem"].tolerances.filling_residual == 2e-3
    assert called["problem"].integration.num_threads == 3
    assert np.allclose(result.to_tb()[(0,)], np.array([[1.0]]))
    assert result.errors.density_matrix_integration == 0.0
    assert result.mu == 0.0
    assert result.filling == 1.0


def test_adaptive_simplex_scf_passes_required_coordinates_for_dense_hamiltonian(
    monkeypatch,
):
    import meanfi.scf.problem as scf_problem

    model = Model(
        {(0,): np.diag([-0.5, 0.5]).astype(complex)},
        {(0,): np.array([[0.0, 1.0], [1.0, 0.0]])},
        filling=1.0,
    )
    required = model.required_coordinates
    captured = {}

    def fake_density_update(problem, **kwargs):
        captured["density_coordinates"] = problem.density_coordinates
        return DensityResult(
            entries=_DensityEntries(
                required,
                np.ones(required.value_count, dtype=complex),
                np.zeros(required.value_count),
            ),
            mu=0.0,
            filling=1.0,
            errors=ErrorValues(
                density_matrix_integration=0.0,
                filling_residual=0.0,
                charge_integration=0.0,
            ),
            statistics=SimpleNamespace(),
            band_energy=0.0,
        )

    monkeypatch.setattr(
        scf_problem,
        "evaluate_density",
        fake_density_update,
    )
    problem = SCFProblem(
        model,
        build_density_problem(
            model.h_0,
            kT=model.kT,
            keys=model.required_coordinates.keys,
            integration=FermiSimplex(),
            tolerances=default_solver_tolerances(1e-3),
            density_coordinates=required,
        ),
    )

    problem.evaluate_state(
        ActiveDensityState(model._space, np.zeros(model._space.num_params)),
        0.0,
    )

    assert captured["density_coordinates"] is required


def test_adaptive_simplex_empty_density_selection_reports_no_density_call(monkeypatch):
    import meanfi.density.integrate.simplex as simplex_integration
    from meanfi.density.integrate.simplex import mesh as native_mesh

    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,)],
        pairs_by_key={
            (0,): (
                np.array([], dtype=int),
                np.array([], dtype=int),
            )
        },
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
        error_stats=SimpleNamespace(
            hamiltonian_evaluations=3,
            full_eigensystems=0,
            reduced_eigensystems=0,
            norm_eigensystems=0,
        ),
    )

    monkeypatch.setattr(native_mesh, "_spectral_mesh", lambda h, **kwargs: mesh)
    monkeypatch.setattr(
        simplex_integration, "_zero_temperature_entropy", lambda mesh, mu: 0.0
    )
    monkeypatch.setattr(
        simplex_integration, "_occupied_band_energy", lambda mesh, mu: -1.0
    )
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
        native_mesh,
        "_integrate_charge",
        lambda *args, **kwargs: charge_result,
    )

    result = evaluate_density(
        build_density_problem(
            spinful_chain(),
            kT=0.0,
            keys=[(0,)],
            density_coordinates=coordinates,
            integration=FermiSimplex(nk=None, max_refinements=None, num_threads=None),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=0.001,
                charge_integration=0.001,
                filling_residual=0.001,
            ),
        ),
        filling=1.0,
        mu_guess=0.0,
        mu_tol=1e-10,
        max_charge_evaluations=None,
    )

    assert result.mu == 0.0
    assert result.values.size == result.entry_errors.size == 0
    assert result.statistics.charge_integration_calls == 1
    assert result.statistics.density_integration_calls == 0


def test_adaptive_simplex_calls_fermisimplex_density_api_with_preview_depth_one():
    from meanfi.density.integrate.simplex import mesh as native_mesh

    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,), (1,)],
        pairs_by_key={
            (0,): (np.array([0]), np.array([1])),
            (1,): (np.array([1]), np.array([0])),
        },
    )
    calls = []

    class Mesh:
        def integrate_density_components(self, **kwargs):
            calls.append(kwargs)
            return "density"

    result = native_mesh._integrate_density(
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
    import meanfi.density.integrate.simplex.mesh as simplex_integration

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


@pytest.mark.parametrize("ndim", (3, 4))
def test_zero_temperature_backend_supports_higher_dimensions(ndim):
    key = (0,) * ndim
    result = density_matrix_at_mu(
        {key: np.diag([-1.0, 1.0])},
        mu=0.0,
        kT=0.0,
        keys=[key],
        integration=FermiSimplex(density_matrix_tol=1e-12, max_refinements=10),
    )

    assert np.allclose(
        result.to_tb()[key],
        np.diag([1.0, 0.0]),
        atol=1e-12,
    )
    assert result.errors.density_matrix_integration is not None
