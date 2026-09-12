"""Prescribed native simplex meshes and their numerical/result contracts."""

from math import factorial

import numpy as np
import pytest

from meanfi import AdaptiveSimplex, density_matrix, density_matrix_at_mu
from meanfi.density.integrate.simplex import (
    _spectral_mesh,
    density_matrix_at_mu_zero_temp,
    density_matrix_zero_temp,
)
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
    import meanfi.density.integrate.simplex as simplex

    def unexpected_construction(*args, **kwargs):
        pytest.fail("mesh allocation must follow the limit check")

    monkeypatch.setattr(simplex, "SpectralMesh", unexpected_construction)
    with pytest.raises(RuntimeError, match=r"1089.*nk=1000.*max_points=1000"):
        _spectral_mesh({(0, 0): np.eye(1)}, nk=1000, max_points=1000)


def test_prescribed_density_uses_simplex_rule_without_refinement_or_previews():
    h = _chain()
    mu = 0.2
    rho, error, info = density_matrix_at_mu_zero_temp(
        h,
        mu=mu,
        keys=[(0,), (1,)],
        density_atol=1e-20,
        density_rtol=0.0,
        nk=100,
        max_points=129,
        max_subdivisions=0,
    )
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
    assert info.subdivisions == 0


def test_prescribed_filling_reuses_native_spectra_and_preserves_band_energy():
    rho, error, mu, info = density_matrix_zero_temp(
        _chain(),
        filling=0.5,
        keys=[(0,), (1,)],
        charge_tol=1e-20,
        filling_tol=1e-10,
        density_atol=1e-20,
        density_rtol=0.0,
        mu_guess=0.2,
        mu_xtol=1e-12,
        max_charge_evaluations=40,
        nk=100,
        max_points=129,
        max_subdivisions=0,
        include_band_energy=True,
    )
    assert abs(mu) < 1e-10
    np.testing.assert_allclose(rho[(0,)], 0.5, atol=1e-10)
    assert abs(info.band_energy + 2 / np.pi) < 2e-4
    assert error is None
    assert info.charge_error is None
    assert info.n_diagonalizations == info.n_kpoints == 129
    assert info.charge_evaluations > 1
    assert info.density_n_kernel_evals == 0
    assert info.subdivisions == 0


def test_prescribed_simplex_public_tolerance_keeps_fixed_mode_and_selected_layout():
    h = _chain()
    integration = AdaptiveSimplex(nk=65, max_refinements=0)
    coordinates = DensityCoordinates.from_entries(
        size=1,
        keys=[(1,)],
        entries=(((1,), 0, 0),),
        allow_empty=False,
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
        size=1, keys=[(1,)], entries=(((1,), 0, 0),), allow_empty=False
    )
    _, error, info = density_matrix_at_mu_zero_temp(
        _chain(),
        mu=0.2,
        keys=[(1,)],
        density_coordinates=coordinates,
        density_atol=1e-5,
        density_rtol=0.0,
        nk=129,
        max_points=129,
    )
    assert abs(info.charge - (1 - np.arccos(0.1) / np.pi)) < 2e-5
    assert info.n_diagonalizations == 129
    assert error is None


def test_empty_fixed_mu_simplex_still_evaluates_charge():
    coordinates = DensityCoordinates.from_entries(
        size=1, keys=[(1,)], entries=(), allow_empty=True
    )
    _, error, info = density_matrix_at_mu_zero_temp(
        _chain(),
        mu=0.2,
        keys=[(1,)],
        density_coordinates=coordinates,
        density_atol=1e-5,
        density_rtol=0.0,
        nk=129,
        max_points=129,
    )
    assert abs(info.charge - (1 - np.arccos(0.1) / np.pi)) < 2e-5
    assert info.n_diagonalizations == info.n_cached_nodes == 129
    assert error is None


def test_adaptive_density_refinement_keeps_root_consistent_with_final_native_mesh(
    monkeypatch,
):
    import meanfi.density.integrate.simplex as simplex

    meshes = []
    construct = simplex._spectral_mesh

    def record_mesh(*args, **kwargs):
        mesh = construct(*args, **kwargs)
        meshes.append(mesh)
        return mesh

    monkeypatch.setattr(simplex, "_spectral_mesh", record_mesh)
    rho, error, mu, info = density_matrix_zero_temp(
        _chain(),
        filling=0.4,
        keys=[(0,), (1,)],
        charge_tol=1e-4,
        filling_tol=1e-10,
        density_atol=1e-4,
        density_rtol=0.0,
        mu_guess=0.0,
        mu_xtol=1e-12,
        max_charge_evaluations=100,
        max_points=10000,
    )
    final_charge = meshes[0].estimate_charge_on_current_mesh(mu=mu).value
    assert abs(final_charge - 0.4) <= 1e-10
    assert info.charge == final_charge
    assert abs(rho[(0,)][0, 0] - final_charge) <= error[(0,)][0, 0]
    assert info.charge_error <= 1e-4
    assert info.density_integration_calls >= 2


@pytest.mark.parametrize("empty", [False, True])
def test_adaptive_fixed_mu_enforces_charge_target_for_selected_layouts(
    monkeypatch, empty
):
    import meanfi.density.integrate.simplex as simplex

    meshes = []
    construct = simplex._spectral_mesh

    def record_mesh(*args, **kwargs):
        mesh = construct(*args, **kwargs)
        meshes.append(mesh)
        return mesh

    monkeypatch.setattr(simplex, "_spectral_mesh", record_mesh)
    coordinates = DensityCoordinates.from_entries(
        size=1, keys=[(1,)], entries=() if empty else (((1,), 0, 0),), allow_empty=empty
    )
    _, _, info = density_matrix_at_mu_zero_temp(
        _chain(),
        mu=0.2,
        keys=[(1,)],
        density_coordinates=coordinates,
        density_atol=0.1,
        density_rtol=0.0,
        charge_tol=1e-7,
        max_points=10000,
    )
    check = meshes[0].integrate_charge(mu=0.2, target_error=1e-7, max_refinements=0)
    assert check.stats.refinements == 0
    assert check.stopping_error == pytest.approx(info.charge_error, rel=1e-8)
    assert info.charge_error <= 1e-7
    assert info.charge == pytest.approx(check.value, abs=1e-14)
    assert abs(info.charge - (1 - np.arccos(0.1) / np.pi)) < 2e-7
    assert info.n_diagonalizations > info.n_kpoints
    assert info.subdivisions > 0


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
    coordinates = DensityCoordinates.from_entries(
        size=1, keys=[(1,)], entries=(), allow_empty=True
    )
    with pytest.raises(RuntimeError, match="did not converge.*max_points"):
        density_matrix_at_mu_zero_temp(
            _chain(),
            mu=0.2,
            keys=[(1,)],
            density_coordinates=coordinates,
            density_atol=1.0,
            density_rtol=0.0,
            charge_tol=1e-7,
            max_points=3,
        )
