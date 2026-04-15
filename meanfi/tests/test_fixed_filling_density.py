import numpy as np

from meanfi import add_tb, density_matrix, fermi_dirac, meanfield


def _hubbard_chain_hamiltonian(U=2.0):
    hopping = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
    h_0 = {(0,): hopping + hopping.T.conj(), (1,): hopping, (-1,): hopping.T.conj()}
    h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
    rho_trial = {(0,): np.diag([0.7, 0.3, 0.3, 0.7])}
    return add_tb(h_0, meanfield(rho_trial, h_int))


def _local_spinful_2d(energy=1.0):
    return {(0, 0): np.diag([-energy, energy])}


def test_fixed_filling_density_reports_separate_charge_and_density_tolerances():
    tb = _hubbard_chain_hamiltonian()
    rho, error, mu, info = density_matrix(
        tb,
        filling=2.0,
        kT=0.1,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-6,
        density_atol=1e-8,
        density_rtol=0.0,
    )

    assert np.isfinite(mu)
    assert abs(info.charge - 2.0) <= 1e-6
    assert info.charge_integral_atol == 2.5e-7
    assert info.density_atol == 1e-8
    assert info.density_rtol == 0.0
    assert info.density_integration_calls == 1
    assert info.n_kernel_evals == info.charge_n_kernel_evals + info.density_n_kernel_evals
    assert set(rho) == {(0,), (1,), (-1,)}
    assert set(error) == {(0,), (1,), (-1,)}


def test_fixed_filling_density_matches_dense_reference_in_2d():
    tb = _local_spinful_2d()
    filling = 1.0
    kT = 0.2
    keys = [(0, 0), (1, 0), (0, 1)]

    rho, _, mu, info = density_matrix(
        tb,
        filling=filling,
        kT=kT,
        keys=keys,
        charge_tol=1e-9,
        density_atol=1e-8,
    )
    occupations = fermi_dirac(np.array([-1.0, 1.0]), kT, 0.0)
    rho_expected = np.diag(occupations)

    assert abs(mu) < 5e-6
    assert abs(info.charge - filling) < 1e-9
    assert np.allclose(rho[(0, 0)], rho_expected, atol=5e-7)
    assert np.allclose(rho[(1, 0)], np.zeros((2, 2)), atol=5e-7)
    assert np.allclose(rho[(0, 1)], np.zeros((2, 2)), atol=5e-7)
