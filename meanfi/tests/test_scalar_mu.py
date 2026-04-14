import numpy as np
from scipy.integrate import cubature
from scipy.optimize import root_scalar

from meanfi import density_matrix, density_matrix_at_mu, fermi_dirac
from meanfi.tb.transforms import tb_to_kfunc


def _spinful_chain(t=1.0):
    hopping = -t * np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


def _charge_from_cubature(tb, mu, kT, atol=1e-7):
    hkfunc = tb_to_kfunc(tb)

    def integrand(k):
        eigenvalues = np.linalg.eigvalsh(hkfunc(k))
        occupation = fermi_dirac(eigenvalues, kT, mu)
        return np.sum(occupation, axis=-1, keepdims=True) / (2 * np.pi)

    result = cubature(integrand, np.array([-np.pi]), np.array([np.pi]), atol=atol)
    if result.status != "converged":
        raise ValueError("cubature failed")
    return float(result.estimate[0])


def _solve_mu_with_cubature(tb, filling, kT, atol=1e-7):
    def residual(mu):
        return _charge_from_cubature(tb, mu=mu, kT=kT, atol=atol) - filling

    bracket = (-10.0, 10.0)
    return float(root_scalar(residual, bracket=bracket, method="brentq").root)


def _density_from_dense_grid(tb, mu, kT, keys, nk=20001):
    hkfunc = tb_to_kfunc(tb)
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    h_k = hkfunc(ks[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(h_k)
    occupation = fermi_dirac(eigenvalues, kT, mu)
    density_matrix_k = (
        eigenvectors
        * occupation[:, np.newaxis, :]
        @ eigenvectors.conj().transpose(0, 2, 1)
    )

    rho = {}
    for key in keys:
        phase = np.exp(1j * ks * key[0])
        rho[key] = np.einsum("k,kab->ab", phase / nk, density_matrix_k)
    return rho


def test_density_matrix_at_mu_matches_dense_reference():
    tb = _spinful_chain()
    mu = 0.0
    kT = 0.15
    keys = [(0,), (1,), (-1,)]

    rho_stateful, _, info = density_matrix_at_mu(
        tb, mu=mu, kT=kT, keys=keys, density_atol=1e-8
    )
    rho_dense = _density_from_dense_grid(tb, mu=mu, kT=kT, keys=keys)

    assert info.n_kernel_evals > 0
    for key in keys:
        assert np.allclose(rho_stateful[key], rho_dense[key], atol=5e-5)


def test_density_matrix_matches_cubature_mu_and_dense_density():
    tb = _spinful_chain()
    filling = 0.7
    kT = 0.15
    keys = [(0,), (1,), (-1,)]

    rho_stateful, _, mu_stateful, info = density_matrix(
        tb, filling=filling, kT=kT, keys=keys, charge_tol=1e-8, density_atol=1e-8
    )
    mu_reference = _solve_mu_with_cubature(tb, filling=filling, kT=kT)
    rho_dense = _density_from_dense_grid(tb, mu=mu_reference, kT=kT, keys=keys)

    assert abs(mu_stateful - mu_reference) < 5e-6
    assert abs(info.charge - filling) < 1e-8
    assert info.charge_error <= info.charge_integral_atol
    for key in keys:
        assert np.allclose(rho_stateful[key], rho_dense[key], atol=5e-5)


def test_density_matrix_half_filling_keeps_particle_hole_symmetry():
    tb = _spinful_chain()
    _, _, mu, info = density_matrix(
        tb, filling=1.0, kT=0.2, keys=[(0,)], charge_tol=1e-9, density_atol=1e-8
    )

    assert abs(mu) < 5e-7
    assert abs(info.charge - 1.0) < 1e-9
