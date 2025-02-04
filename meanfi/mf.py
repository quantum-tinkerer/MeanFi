import numpy as np
from typing import Tuple

from meanfi.tb.tb import add_tb, _tb_type
from meanfi.tb.transforms import tb_to_kfunc

from scipy.integrate import cubature

def complex_cubature(integrand, a, b, args=(), cubature_kwargs={'atol' : 1e-6}):
    """
    Integrate a complex-valued function using scipy.integrate.cubature.

    Parameters
    ----------
    integrand :
        Complex-valued function to integrate.
    a :
        Lower integration limit.
    b :
        Upper integration limit.
    args :
        Additional arguments to pass to the integrand.
    cubature_kwargs :
        Additional keyword arguments to pass to scipy.integrate.cubature.

    Returns
    -------
    :
        Complex-valued integral and error estimate.
    """

    def real_integrand(k, *args):
        return integrand(k, *args).real
    def imag_integrand(k, *args):
        return integrand(k, *args).imag

    real_result = cubature(real_integrand, a, b, args=args, **cubature_kwargs)
    imag_result = cubature(imag_integrand, a, b, args=args, **cubature_kwargs)

    real_integral = real_result.estimate
    imag_integral = imag_result.estimate

    real_error = real_result.error
    imag_error = imag_result.error

    if (real_result.status and imag_result.status) == 'converged':
        return real_integral + 1j * imag_integral, real_error + 1j * imag_error
    else:
        raise ValueError('Integration did not converge')

def density_matrix(h: _tb_type, mu: float, beta : float, keys : list, atol=1e-5) -> Tuple[_tb_type, float]:
    """Compute the real-space density matrix tight-binding dictionary.

    Parameters
    ----------
    h :
        Hamiltonian tight-binding dictionary from which to construct the density matrix.
    mu :
        Number of particles in a unit cell.
        Used to determine the Fermi level.
    beta :
        Inverse temperature.
    keys :
        List of keys to compute the density matrix for.

    Returns
    -------
    :
        Density matrix tight-binding dictionary
    """
    def density_matrix_k(H_k, mu, beta=1e2):
        eigenvalues, U = np.linalg.eigh(H_k)
        fermi_distribution = 1.0 / (1.0 + np.exp(beta * (eigenvalues - mu)))
        density_matrix = U * fermi_distribution[:, None, :] @ U.conj().transpose(0, 2, 1)
        return density_matrix
    
    hkfunc = tb_to_kfunc(h)
    def integrand(k, mu, beta, key):
        H = hkfunc(k)
        return np.exp(1j * np.dot(k, key))[:, np.newaxis, np.newaxis] * density_matrix_k(H, mu=mu, beta=beta) / (4*np.pi**2)
    
    density_matrix_dict = {}
    error_dict = {}
    for key in keys:
        rho, error = complex_cubature(integrand, [-np.pi, -np.pi], [np.pi, np.pi], args=(mu, beta, np.array(key, dtype=float)), cubature_kwargs={'atol' : atol})
        density_matrix_dict[key] = rho
        error_dict[key] = error
    return density_matrix_dict, error_dict


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the mean-field correction from the density matrix.

    Parameters
    ----------
    density_matrix :
        Density matrix tight-binding dictionary.
    h_int :
        Interaction hermitian Hamiltonian tight-binding dictionary.
    Returns
    -------
    :
        Mean-field correction tight-binding dictionary.

    Notes
    -----

    The interaction h_int must be of density-density type.
    For example, h_int[(1,)][i, j] = V means a repulsive interaction
    of strength V between two particles with internal degrees of freedom i and j
    separated by 1 lattice vector.
    """
    n = len(list(density_matrix)[0])
    local_key = tuple(np.zeros((n,), dtype=int))

    direct = {
        local_key: np.sum(
            np.array(
                [
                    np.diag(
                        np.einsum("pp,pn->n", density_matrix[local_key], h_int[vec])
                    )
                    for vec in frozenset(h_int)
                ]
            ),
            axis=0,
        )
    }

    exchange = {
        vec: -1 * h_int.get(vec, 0) * density_matrix[vec] for vec in frozenset(h_int)
    }
    return add_tb(direct, exchange)


def fermi_on_kgrid(vals: np.ndarray, filling: float) -> float:
    """Compute the Fermi energy on a grid of k-points.

    Parameters
    ----------
    vals :
        Eigenvalues of a hamiltonian sampled on a k-point grid with shape (nk, nk, ..., ndof, ndof),
        where ndof is number of internal degrees of freedom.
    filling :
        Number of particles in a unit cell.
        Used to determine the Fermi level.
    Returns
    -------
    :
        Fermi energy
    """
    norbs = vals.shape[-1]
    vals_flat = np.sort(vals.flatten())
    ne = len(vals_flat)
    ifermi = int(round(ne * filling / norbs))
    if ifermi >= ne:
        return vals_flat[-1]
    elif ifermi == 0:
        return vals_flat[0]
    else:
        fermi = (vals_flat[ifermi - 1] + vals_flat[ifermi]) / 2
        return fermi
