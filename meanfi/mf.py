import numpy as np
from typing import Tuple

from meanfi.tb.tb import add_tb, _tb_type
from meanfi.tb.transforms import tb_to_kfunc

from scipy.integrate import cubature

def fermi_dirac(E: np.ndarray, kT: float, fermi: float) -> np.ndarray:
    """
    Calculate the value of the Fermi-Dirac distribution at energy `E` and temperature `T`.

    Parameters
    ----------
    `E: np.ndarray(float)` :
        The energy at which to find the value of the distribution. Can also be an array of values.
    `kT: float` :
        The temperature in Kelvin and Boltzmann constant.
    `fermi: float` :
        The Fermi level.

    Returns
    -------
        The value of the Fermi-Dirac distribution.
    """
    if kT == 0:
        fd = E < fermi
        return fd
    else:
        fd = np.empty_like(E, dtype=float)
        exponent = (E - fermi) / kT
        sign_mask = (
            E >= fermi
        )  # Holds the indices for all positive values of the exponent.
        
        # Precalculating the two options.
        pos_exp = np.exp(-exponent[sign_mask])
        neg_exp = np.exp(exponent[~sign_mask])

        fd[sign_mask] = pos_exp / (pos_exp + 1)
        fd[~sign_mask] = 1 / (neg_exp + 1)

        return fd

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
    def complex_integrand(k, *args):
        value = integrand(k, *args)
        value_real = value.real
        value_imag = value.imag
        return np.stack((value_real, value_imag), axis=-1, dtype=float)

    result = cubature(complex_integrand, a, b, args=args, **cubature_kwargs)
    
    if result.status == 'converged':
        integral_unpacked = result.estimate
        error_unpacked = result.error

        integral_real = integral_unpacked[..., 0]
        integral_imag = integral_unpacked[..., 1]

        error_real = error_unpacked[..., 0]
        error_imag = error_unpacked[..., 1]

        return integral_real + 1j * integral_imag, error_real + 1j * error_imag
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
    ndim = len(keys[0])

    def density_matrix_k(H_k, mu, beta=1e2):
        eigenvalues, U = np.linalg.eigh(H_k)
        fermi_distribution = 1.0 / (1.0 + np.exp(beta * (eigenvalues - mu)))
        density_matrix = U * fermi_distribution[:, None, :] @ U.conj().transpose(0, 2, 1)
        return density_matrix
    
    hkfunc = tb_to_kfunc(h)
    def integrand(k, mu, beta, keys):
        dm_k = density_matrix_k(hkfunc(k), mu=mu, beta=beta)
        phase = np.exp(1j * np.dot(k, keys.T))
        return dm_k[..., np.newaxis] * phase[:, np.newaxis, np.newaxis, :] / (2*np.pi)**ndim

    density_matrix_dict = {}
    error_dict = {}
    
    bounds_lower  = np.array([-np.pi] * ndim)
    bounds_upper = np.array([np.pi] * ndim)
    rho, error = complex_cubature(integrand, bounds_lower, bounds_upper, args=(mu, beta, np.array(keys, dtype=float)), cubature_kwargs={'atol' : atol})
    for idx, key in enumerate(keys):
        density_matrix_dict[key] = rho[..., idx]
        error_dict[key] = error[..., idx]
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
