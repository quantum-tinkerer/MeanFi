import numpy as np
import pytest

from meanfi import add_tb, density_matrix, density_matrix_at_mu, fermi_dirac
from meanfi.zero_temp import _NATIVE_ZERO_TEMP_AVAILABLE
from meanfi.tests.helpers import (
    antiferromagnetic_guess,
    bipartite_hubbard_1d,
    duplicated_local_two_band_1d,
    exact_spinful_chain_charge,
    exact_spinful_chain_mu,
    spinful_chain,
)


pytestmark = pytest.mark.physics
requires_native = pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)


def test_zero_dimensional_density_matrix_at_mu_matches_exact_occupation():
    h = {(): np.diag([-1.0, 2.0])}
    mu = 0.3
    kT = 0.4

    rho, error, info = density_matrix_at_mu(h, mu=mu, kT=kT, keys=[()])
    expected = np.diag(fermi_dirac(np.array([-1.0, 2.0]), kT, mu))

    assert np.allclose(rho[()], expected, atol=1e-12)
    assert np.allclose(error[()], np.zeros((2, 2)), atol=0.0)
    assert info.n_kernel_evals == 1
    assert info.n_evaluator_evals == 1


def test_density_matrix_respects_hermiticity_and_charge_sum_rule():
    h_0, _h_int = bipartite_hubbard_1d(U=4.0)
    h_trial = add_tb(h_0, antiferromagnetic_guess(0.7, 1))

    rho, _, _, info = density_matrix(
        h_trial,
        filling=2.0,
        kT=0.1,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-8,
        density_atol=1e-8,
    )

    assert np.allclose(rho[(0,)], rho[(0,)].conj().T, atol=1e-8)
    assert np.allclose(rho[(-1,)], rho[(1,)].conj().T, atol=1e-8)
    assert abs(np.trace(rho[(0,)]).real - 2.0) <= 1e-8
    assert abs(info.charge - 2.0) <= 1e-8


def test_half_filling_keeps_particle_hole_symmetry():
    _rho, _error, mu, info = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.2,
        keys=[(0,)],
        charge_tol=1e-9,
        density_atol=1e-8,
    )

    assert abs(mu) < 5e-7
    assert abs(info.charge - 1.0) < 1e-9


@requires_native
def test_zero_temperature_fixed_filling_tracks_exact_mu_on_analytic_chain():
    tb = spinful_chain()
    for filling in (0.1, 0.25, 0.5, 0.9, 1.1, 1.5, 1.75, 1.9):
        _rho, _error, mu, info = density_matrix(
            tb,
            filling=filling,
            kT=0.0,
            keys=[(0,), (1,)],
            charge_tol=5e-5,
            density_atol=1e-2,
            max_subdivisions=600,
        )

        assert abs(mu - exact_spinful_chain_mu(filling)) <= 1e-4
        assert abs(exact_spinful_chain_charge(mu) - filling) <= 3e-5
        assert info.subdivisions > 0


@requires_native
def test_zero_temperature_density_is_invariant_under_equivalent_local_supercell():
    primitive, doubled = duplicated_local_two_band_1d()

    rho_primitive, _, mu_primitive, info_primitive = density_matrix(
        primitive,
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        charge_tol=1e-12,
        density_atol=1e-12,
        max_subdivisions=4,
    )
    rho_doubled, _, mu_doubled, info_doubled = density_matrix(
        doubled,
        filling=2.0,
        kT=0.0,
        keys=[(0,)],
        charge_tol=1e-12,
        density_atol=1e-12,
        max_subdivisions=4,
    )

    assert info_primitive.error_estimate_available is True
    assert info_doubled.error_estimate_available is True
    assert info_primitive.subdivisions == info_doubled.subdivisions == 0
    assert abs(mu_primitive - mu_doubled) < 1e-12
    assert np.allclose(rho_primitive[(0,)], rho_doubled[(0,)][:2, :2], atol=1e-12)
    assert np.allclose(rho_primitive[(0,)], rho_doubled[(0,)][2:, 2:], atol=1e-12)
    assert np.allclose(rho_doubled[(0,)][:2, 2:], np.zeros((2, 2)), atol=1e-12)

