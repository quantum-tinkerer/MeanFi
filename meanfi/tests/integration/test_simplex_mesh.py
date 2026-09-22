"""Prescribed native simplex meshes and their numerical/result contracts."""

from dataclasses import replace
from meanfi import default_solver_tolerances
from meanfi.density.problem import build_density_problem
from meanfi.density.density import evaluate_density


from math import factorial

import numpy as np
import pytest

from meanfi import BlochHamiltonian, FermiSimplex, density_matrix, density_matrix_at_mu
from meanfi.density.integrate.simplex.mesh import _spectral_mesh
from meanfi.space.coordinates import DensityCoordinates


pytestmark = pytest.mark.integration


def _chain():
    return {
        (0,): np.array([[0.0]]),
        (1,): np.array([[1.0]]),
        (-1,): np.array([[1.0]]),
    }


@pytest.mark.parametrize(
    "dimension,nk,actual",
    [(1, 1, 2), (1, 100, 129), (2, 1000, 1089), (2, 4096, 4225), (3, 4096, 4913)],
)
def test_native_prescribed_mesh_rounds_total_nodes_and_counts_both_boundaries(
    dimension, nk, actual
):
    mesh = _spectral_mesh({(0,) * dimension: np.eye(1)}, nk=nk, max_points=actual)
    assert mesh.active_vertices == actual
    assert mesh.active_simplices == factorial(dimension) * 2 ** (
        dimension * mesh.root_level
    )
    np.testing.assert_array_equal(mesh.points.min(axis=0), np.zeros(dimension))
    np.testing.assert_array_equal(mesh.points.max(axis=0), np.ones(dimension))
    assert mesh.cached_vertices == 0


def test_prescribed_simplex_rounding_obeys_hard_limit_before_native_allocation(
    monkeypatch,
):
    import meanfi.density.integrate.simplex.mesh as simplex

    def unexpected_construction(*args, **kwargs):
        pytest.fail("mesh allocation must follow the limit check")

    monkeypatch.setattr(simplex, "SpectralMesh", unexpected_construction)
    with pytest.raises(RuntimeError, match=r"1089.*nk=1000.*max_points=1000"):
        _spectral_mesh({(0, 0): np.eye(1)}, nk=1000, max_points=1000)


def test_prescribed_density_uses_simplex_rule_without_refinement_or_previews():
    h = _chain()
    mu = 0.2
    result = evaluate_density(
        build_density_problem(
            h,
            kT=0.0,
            keys=[(0,), (1,)],
            density_coordinates=None,
            integration=FermiSimplex(nk=100, max_refinements=0, max_points=129),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-20,
                charge_integration=1e-20,
            ),
        ),
        mu=mu,
    )
    info = result.statistics
    rho = result.to_tb()
    error = result.entry_errors
    mesh = _spectral_mesh(h, nk=100)
    charge = mesh.estimate_charge_on_current_mesh(mu=mu)
    weights = mesh.occupied_weights(mu)[:, 0]
    reference_hopping = np.sum(weights * np.exp(2j * np.pi * mesh.points[:, 0]))
    np.testing.assert_allclose(rho[(0,)][0, 0], charge.value, atol=1e-14)
    np.testing.assert_allclose(rho[(1,)][0, 0], reference_hopping, atol=1e-14)
    assert abs(charge.value - (1 - np.arccos(mu / 2) / np.pi)) < 2e-5
    assert error is None
    assert not info.error_estimate_available
    assert info.requested_nk == 100
    assert info.n_kpoints == info.n_cached_nodes == info.n_diagonalizations == 129
    assert info.refinements == 0


def test_prescribed_filling_reuses_native_spectra_and_preserves_band_energy():
    result = evaluate_density(
        build_density_problem(
            _chain(),
            kT=0.0,
            keys=[(0,), (1,)],
            density_coordinates=None,
            integration=FermiSimplex(nk=100, max_refinements=0, max_points=129),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-20,
                charge_integration=1e-20,
                filling_residual=1e-10,
                mu_tol=1e-12,
            ),
        ),
        filling=0.5,
        mu_guess=0.2,
        max_charge_evaluations=40,
    )
    info = result.statistics
    rho = result.to_tb()
    error = result.entry_errors
    mu = result.mu
    assert abs(mu) < 1e-10
    np.testing.assert_allclose(rho[(0,)], 0.5, atol=1e-10)
    assert abs(result.band_energy + 2 / np.pi) < 2e-4
    assert error is None
    assert result.errors.charge_integration is None
    assert info.n_kpoints == 129
    assert info.n_energy_evaluations == 128
    assert info.n_diagonalizations == info.n_kpoints + info.n_energy_evaluations
    assert info.charge_evaluations > 1
    assert info.refinements == 0


def test_prescribed_simplex_public_tolerance_keeps_fixed_mode_and_selected_layout():
    h = _chain()
    integration = FermiSimplex(nk=65, max_refinements=0)
    coordinates = DensityCoordinates.from_entries(
        size=1,
        keys=[(1,)],
        entries=(((1,), 0, 0),),
    )
    full = density_matrix(
        h, filling=0.4, keys=[(0,), (1,)], integration=integration, tol=1e-8
    )
    selected = density_matrix(
        h,
        filling=0.4,
        coordinates=coordinates,
        integration=integration,
        tol=1e-12,
    )
    np.testing.assert_allclose(selected.values, full.values_for(coordinates))
    assert full.errors.density_matrix_integration is None
    assert full.errors.charge_integration is None
    assert full.errors.filling_residual <= 1e-9
    assert selected.errors.density_matrix_integration is None


def test_adaptive_simplex_charge_mesh_storage_limit_is_checked():
    with pytest.raises(RuntimeError, match="max_points"):
        density_matrix_at_mu(
            _chain(),
            mu=0.2,
            keys=[(0,), (1,)],
            integration=FermiSimplex(initial_nk=3, max_points=3),
            tol=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-3,
                charge_integration=1.0,
            ),
        )


def test_selected_fixed_mu_simplex_leaves_filling_unknown_without_diagonal_entries():
    coordinates = DensityCoordinates.from_entries(
        size=1, keys=[(1,)], entries=(((1,), 0, 0),)
    )
    result = evaluate_density(
        build_density_problem(
            _chain(),
            kT=0.0,
            keys=[(1,)],
            density_coordinates=coordinates,
            integration=FermiSimplex(nk=129, max_points=129),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-05,
                charge_integration=1e-05,
            ),
        ),
        mu=0.2,
    )
    info = result.statistics
    error = result.entry_errors
    assert result.filling is None
    assert info.n_diagonalizations == 129
    assert error is None


def test_density_refinement_preserves_charge_stage_without_rechecking(monkeypatch):
    from meanfi.density.integrate.simplex.mesh import SimplexEvaluator

    charge_samples = []
    charge = SimplexEvaluator.charge

    def record_charge(self, mu, *, adaptive):
        assert self.work.density_calls == 0, "charge must finish before density"
        sample = charge(self, mu, adaptive=adaptive)
        if adaptive:
            charge_samples.append((mu, sample.value, sample.stopping_error))
        return sample

    monkeypatch.setattr(SimplexEvaluator, "charge", record_charge)
    result = evaluate_density(
        build_density_problem(
            _chain(),
            kT=0.0,
            keys=[(0,), (1,)],
            density_coordinates=None,
            integration=FermiSimplex(max_points=10000),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-4,
                charge_integration=1e-4,
                filling_residual=1e-10,
                mu_tol=1e-12,
            ),
        ),
        filling=0.4,
        mu_guess=0.0,
        max_charge_evaluations=100,
    )
    mu, filling, charge_error = charge_samples[-1]
    assert result.mu == mu
    assert result.filling == filling
    assert result.errors.filling_residual == abs(filling - 0.4)
    assert result.errors.filling_residual <= 1e-10
    assert result.errors.charge_integration == charge_error
    assert result.statistics.density_integration_calls == 1

    rho = result.to_tb()
    # Exact chain integrals at the returned mu, with 2e-4 absolute accuracy.
    np.testing.assert_allclose(
        [rho[(0,)][0, 0], rho[(1,)][0, 0]],
        [1 - np.arccos(mu / 2) / np.pi, -np.sqrt(1 - (mu / 2) ** 2) / np.pi],
        atol=2e-4,
        rtol=0,
    )
    # The projector trace integrates to the same occupation as the charge mesh.
    np.testing.assert_allclose(rho[(0,)][0, 0].real, filling, atol=1e-12)


@pytest.mark.parametrize("empty", [False, True])
def test_fixed_mu_prepares_charge_mesh_only_for_requested_density(monkeypatch, empty):
    from meanfi.density.integrate.simplex.mesh import SimplexEvaluator

    calls = []
    original_charge = SimplexEvaluator.charge

    def record_charge(self, mu, *, adaptive, target_error=None):
        calls.append((mu, adaptive, target_error))
        return original_charge(self, mu, adaptive=adaptive, target_error=target_error)

    monkeypatch.setattr(SimplexEvaluator, "charge", record_charge)
    coordinates = DensityCoordinates.from_entries(
        size=1, keys=[(1,)], entries=() if empty else (((1,), 0, 0),)
    )
    result = density_matrix_at_mu(
        _chain(),
        mu=0.2,
        coordinates=coordinates,
        integration=FermiSimplex(max_points=10000),
        tol=replace(default_solver_tolerances(1e-3), charge_integration=1e-15),
    )
    assert result.filling is None
    assert result.errors.charge_integration is None
    assert result.errors.filling_residual is None
    assert result.statistics.charge_integration_calls == (0 if empty else 1)
    assert calls == (
        []
        if empty
        else [(0.2, True, default_solver_tolerances(1e-3).density_matrix_integration)]
    )
    assert result.statistics.density_integration_calls == (0 if empty else 1)
    if empty:
        assert result.statistics.n_diagonalizations == 0
    else:
        assert abs(result.values[0] + np.sqrt(0.99) / np.pi) < 2e-4


def test_fixed_mu_filling_uses_available_density_trace():
    result = density_matrix_at_mu(_chain(), mu=0.2, keys=[(0,)])
    assert result.filling == result.to_tb()[(0,)][0, 0].real
    assert result.errors.charge_integration is None
    assert result.statistics.charge_integration_calls == 1
    assert abs(result.filling - (1 - np.arccos(0.1) / np.pi)) < 2e-4


def test_selected_simplex_density_does_not_assemble_unrequested_matrix_entries(
    monkeypatch,
):
    h = {
        (0,): np.array([[0.4, 0.3], [0.3, -0.4]]),
        (1,): -0.7 * np.eye(2),
        (-1,): -0.7 * np.eye(2),
    }
    integration = FermiSimplex(nk=65)
    reference = density_matrix(h, filling=0.8, keys=[(1,)], integration=integration)
    coordinates = DensityCoordinates.from_entries(
        size=2, keys=[(1,)], entries=(((1,), 0, 1),)
    )

    def unexpected_matrix_assembly(*args, **kwargs):
        pytest.fail("selected density must remain coordinate values")

    monkeypatch.setattr("meanfi.results._assemble_blocks", unexpected_matrix_assembly)
    selected = density_matrix(
        h, filling=0.8, coordinates=coordinates, integration=integration
    )
    np.testing.assert_allclose(
        selected.values, reference.values_for(coordinates), atol=1e-12
    )
    assert selected.coordinates is coordinates
    assert selected.filling == pytest.approx(0.8, abs=1e-8)


def test_density_degree_cap_requires_an_odd_rule_from_three():
    assert FermiSimplex().density_max_degree == 7
    with pytest.raises(ValueError, match="density_max_degree"):
        FermiSimplex(density_max_degree=2)
    assert FermiSimplex(density_max_degree=3).density_max_degree == 3


def test_adaptive_density_hp_bisects_without_changing_charge_filling():
    mu = 0.37
    h = BlochHamiltonian(lambda k: np.array([[k / (2 * np.pi)]], complex))
    result = density_matrix_at_mu(
        h,
        mu=mu,
        keys=[(0,), (1,)],
        integration=FermiSimplex(
            initial_nk=3, density_max_degree=3, max_refinements=400
        ),
        tol=replace(
            default_solver_tolerances(1e-4),
            density_matrix_integration=1e-5,
            charge_integration=1e-5,
        ),
    )
    assert result.statistics.refinements > 0
    assert result.statistics.p_refinements == 0
    assert result.statistics.n_kpoints == 3
    assert result.statistics.n_leaves > 2
    assert result.statistics.charge_integration_calls == 1
    assert result.errors.density_matrix_integration <= 1e-5
    assert result.to_tb()[(0,)][0, 0] == pytest.approx(mu, abs=1e-12)
    exact = np.expm1(2j * np.pi * mu) / (2j * np.pi)
    assert result.to_tb()[(1,)][0, 0] == pytest.approx(exact, abs=1e-5)
