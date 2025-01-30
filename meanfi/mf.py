import numpy as np
from scipy.optimize import minimize

from meanfi.tb.tb import add_tb, _tb_type
from meanfi.tb.transforms import tb_to_kgrid, kgrid_to_tb


def density_matrix_kgrid(
    kham: np.ndarray, filling: float, nk: int, ndim: int, kT: float = 0
) -> tuple[np.ndarray, float]:
    """Calculate density matrix on a k-space grid.

    Parameters
    ----------
    kham :
        Hamiltonian from which to construct the density matrix.
        The hamiltonian is sampled on a grid of k-points and has shape (nk, nk, ..., ndof, ndof),
        where ndof is number of internal degrees of freedom.
    filling :
        Number of particles in a unit cell.
        Used to determine the Fermi level.
    kT :
        Temperature for use in the Fermi-Dirac distribution.

    Returns
    -------
    :
        Density matrix on a k-space grid with shape (nk, nk, ..., ndof, ndof) and Fermi energy.
    """
    vals, vecs = np.linalg.eigh(kham)
    fermi_0 = fermi_on_kgrid(vals, filling)

    full_diag = np.sum(np.moveaxis(vecs.conj(), -1, -2) @ vecs, axis=-2)
    result = minimize(
        trace_difference,
        fermi_0,
        args=(vals, full_diag, kT, filling, nk, ndim),
        method="Nelder-Mead",
        options={"fatol": kT / 2, "xatol": kT / 2},
    )
    opt_fermi = float(result.x)

    occ_distribution = np.sqrt(fermi_dirac(vals, kT, opt_fermi))[..., np.newaxis]
    occ_vecs = vecs
    occ_vecs *= np.moveaxis(occ_distribution, -1, -2)
    _density_matrix_kgrid = occ_vecs @ np.moveaxis(occ_vecs, -1, -2).conj()
    return _density_matrix_kgrid, opt_fermi


def trace_difference(
    fermi: float,
    vals: np.ndarray,
    expectation: np.ndarray,
    kT: float,
    filling: float,
    nk: int,
    ndim: int,
) -> float:
    occ_distribution = fermi_dirac(vals, kT, fermi)
    trace = np.sum(expectation * occ_distribution)

    return np.abs(trace - (filling * nk**ndim))


def density_matrix(
    h: _tb_type, filling: float, nk: int, kT: float = 0
) -> tuple[_tb_type, float]:
    """Compute the real-space density matrix tight-binding dictionary.

    Parameters
    ----------
    h :
        Hamiltonian tight-binding dictionary from which to construct the density matrix.
    filling :
        Number of particles in a unit cell.
        Used to determine the Fermi level.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.
    kT :
        Temperature for use in the Fermi-Dirac distribution.

    Returns
    -------
    :
        Density matrix tight-binding dictionary and Fermi energy.
    """
    ndim = len(list(h)[0])
    if ndim > 0:
        kham = tb_to_kgrid(h, nk=nk)
        _density_matrix_krid, fermi = density_matrix_kgrid(kham, filling, nk, ndim, kT)
        return (
            kgrid_to_tb(_density_matrix_krid),
            fermi,
        )
    else:
        _density_matrix, fermi = density_matrix_kgrid(h[()], filling, nk, ndim, kT)
        return {(): _density_matrix}, fermi


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
        fd = np.empty_like(E)
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
