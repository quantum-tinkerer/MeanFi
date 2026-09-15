"""Prescribed native simplex meshes and their numerical/result contracts."""

from dataclasses import replace
from meanfi import default_solver_tolerances
from meanfi.density.problem import build_density_problem
from meanfi.density.density import evaluate_density


from math import factorial

import numpy as np
import pytest

from meanfi import AdaptiveSimplex, density_matrix, density_matrix_at_mu
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
            integration=AdaptiveSimplex(nk=100, max_refinements=0, max_points=129),
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
            integration=AdaptiveSimplex(nk=100, max_refinements=0, max_points=129),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-20,
                charge_integration=1e-20,
                filling_residual=1e-10,
            ),
        ),
        filling=0.5,
        mu_guess=0.2,
        mu_tol=1e-12,
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
    assert info.n_diagonalizations == info.n_kpoints == 129
    assert info.charge_evaluations > 1
    assert info.refinements == 0


def test_prescribed_simplex_public_tolerance_keeps_fixed_mode_and_selected_layout():
    h = _chain()
    integration = AdaptiveSimplex(nk=65, max_refinements=0)
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


def test_adaptive_simplex_preview_storage_limit_is_checked():
    with pytest.raises(RuntimeError, match="cached/preview nodes.*max_points=3"):
        density_matrix_at_mu(
            _chain(),
            mu=0.2,
            keys=[(0,), (1,)],
            integration=AdaptiveSimplex(density_matrix_tol=1e-3, max_points=3),
        )


def test_selected_fixed_mu_simplex_reports_charge_without_diagonal_entries():
    coordinates = DensityCoordinates.from_entries(
        size=1, keys=[(1,)], entries=(((1,), 0, 0),)
    )
    result = evaluate_density(
        build_density_problem(
            _chain(),
            kT=0.0,
            keys=[(1,)],
            density_coordinates=coordinates,
            integration=AdaptiveSimplex(nk=129, max_points=129),
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
    assert abs(result.filling - (1 - np.arccos(0.1) / np.pi)) < 2e-5
    assert info.n_diagonalizations == 129
    assert error is None


def test_empty_fixed_mu_simplex_still_evaluates_charge():
    coordinates = DensityCoordinates.from_entries(size=1, keys=[(1,)], entries=())
    result = evaluate_density(
        build_density_problem(
            _chain(),
            kT=0.0,
            keys=[(1,)],
            density_coordinates=coordinates,
            integration=AdaptiveSimplex(nk=129, max_points=129),
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
    assert abs(result.filling - (1 - np.arccos(0.1) / np.pi)) < 2e-5
    assert info.n_diagonalizations == info.n_cached_nodes == 129
    assert error is None


def test_adaptive_density_refinement_keeps_root_consistent_with_final_native_mesh(
    monkeypatch,
):
    import meanfi.density.integrate.simplex.mesh as simplex

    meshes = []
    construct = simplex._spectral_mesh

    def record_mesh(*args, **kwargs):
        mesh = construct(*args, **kwargs)
        meshes.append(mesh)
        return mesh

    monkeypatch.setattr(simplex, "_spectral_mesh", record_mesh)
    result = evaluate_density(
        build_density_problem(
            _chain(),
            kT=0.0,
            keys=[(0,), (1,)],
            density_coordinates=None,
            integration=AdaptiveSimplex(nk=None, max_points=10000),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=0.0001,
                charge_integration=0.0001,
                filling_residual=1e-10,
            ),
        ),
        filling=0.4,
        mu_guess=0.0,
        mu_tol=1e-12,
        max_charge_evaluations=100,
    )
    info = result.statistics
    rho = result.to_tb()
    error = result.entry_errors
    mu = result.mu
    final_charge = meshes[0].estimate_charge_on_current_mesh(mu=mu).value
    assert abs(final_charge - 0.4) <= 1e-10
    assert result.filling == final_charge
    assert abs(rho[(0,)][0, 0] - final_charge) <= error[0]
    assert result.errors.charge_integration <= 1e-4
    assert info.density_integration_calls >= 2


@pytest.mark.parametrize("empty", [False, True])
def test_adaptive_fixed_mu_enforces_charge_target_for_selected_layouts(
    monkeypatch, empty
):
    import meanfi.density.integrate.simplex.mesh as simplex

    meshes = []
    construct = simplex._spectral_mesh

    def record_mesh(*args, **kwargs):
        mesh = construct(*args, **kwargs)
        meshes.append(mesh)
        return mesh

    monkeypatch.setattr(simplex, "_spectral_mesh", record_mesh)
    coordinates = DensityCoordinates.from_entries(
        size=1, keys=[(1,)], entries=() if empty else (((1,), 0, 0),)
    )
    result = evaluate_density(
        build_density_problem(
            _chain(),
            kT=0.0,
            keys=[(1,)],
            density_coordinates=coordinates,
            integration=AdaptiveSimplex(nk=None, max_points=10000),
            tolerances=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=0.1,
                charge_integration=1e-07,
            ),
        ),
        mu=0.2,
    )
    info = result.statistics
    check = meshes[0].integrate_charge(mu=0.2, target_error=1e-7, max_refinements=0)
    assert check.stats.refinements == 0
    assert check.stopping_error == pytest.approx(
        result.errors.charge_integration, rel=1e-8
    )
    assert result.errors.charge_integration <= 1e-7
    assert result.filling == pytest.approx(check.value, abs=1e-14)
    assert abs(result.filling - (1 - np.arccos(0.1) / np.pi)) < 2e-7
    assert info.n_diagonalizations > info.n_kpoints
    assert info.refinements > 0


def test_fixed_mu_simplex_charge_target_cannot_be_ignored_at_refinement_limit():
    with pytest.raises(RuntimeError, match="did not converge"):
        density_matrix_at_mu(
            _chain(),
            mu=0.2,
            keys=[(0,)],
            integration=AdaptiveSimplex(
                density_matrix_tol=1.0, charge_tol=1e-12, max_refinements=0
            ),
        )


def test_fixed_mu_simplex_reports_both_integration_errors_publicly():
    result = density_matrix_at_mu(
        _chain(),
        mu=0.2,
        keys=[(1,)],
        integration=AdaptiveSimplex(density_matrix_tol=1e-3, charge_tol=1e-7),
    )
    assert result.errors.density_matrix_integration <= 1e-3
    assert result.errors.charge_integration <= 1e-7
    assert result.errors.filling_residual is None


def test_empty_fixed_mu_charge_target_respects_point_limit():
    coordinates = DensityCoordinates.from_entries(size=1, keys=[(1,)], entries=())
    with pytest.raises(RuntimeError, match="did not converge.*max_points"):
        evaluate_density(
            build_density_problem(
                _chain(),
                kT=0.0,
                keys=[(1,)],
                density_coordinates=coordinates,
                integration=AdaptiveSimplex(nk=None, max_points=3),
                tolerances=replace(
                    default_solver_tolerances(1e-3),
                    density_matrix_integration=1.0,
                    charge_integration=1e-07,
                ),
            ),
            mu=0.2,
        )


def test_selected_simplex_density_does_not_assemble_unrequested_matrix_entries(
    monkeypatch,
):
    h = {
        (0,): np.array([[0.4, 0.3], [0.3, -0.4]]),
        (1,): -0.7 * np.eye(2),
        (-1,): -0.7 * np.eye(2),
    }
    integration = AdaptiveSimplex(nk=65)
    reference = density_matrix(h, filling=0.8, keys=[(1,)], integration=integration)
    coordinates = DensityCoordinates.from_entries(
        size=2, keys=[(1,)], entries=(((1,), 0, 1),)
    )

    def unexpected_matrix_assembly(*args, **kwargs):
        pytest.fail("selected density must remain coordinate values")

    monkeypatch.setattr(DensityCoordinates, "values_to_tb", unexpected_matrix_assembly)
    selected = density_matrix(
        h, filling=0.8, coordinates=coordinates, integration=integration
    )
    np.testing.assert_allclose(
        selected.values, reference.values_for(coordinates), atol=1e-12
    )
    assert selected.coordinates is coordinates
    assert selected.filling == pytest.approx(0.8, abs=1e-8)
