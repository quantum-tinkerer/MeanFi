import numpy as np
from typing import Tuple

from meanfi.tb.tb import add_tb, _tb_type
from meanfi.tb.transforms import tb_to_kgrid, kgrid_to_tb


def density_matrix_kgrid(kham: np.ndarray, filling: float) -> Tuple[np.ndarray, float]:
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

    Returns
    -------
    :
        Density matrix on a k-space grid with shape (nk, nk, ..., ndof, ndof) and Fermi energy.
    """
    vals, vecs = np.linalg.eigh(kham)
    fermi = fermi_on_kgrid(vals, filling)
    unocc_vals = vals > fermi
    occ_vecs = vecs
    np.moveaxis(occ_vecs, -1, -2)[unocc_vals, :] = 0
    _density_matrix_krid = occ_vecs @ np.moveaxis(occ_vecs, -1, -2).conj()
    return _density_matrix_krid, fermi


def density_matrix(h: _tb_type, filling: float, nk: int) -> Tuple[_tb_type, float]:
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

    Returns
    -------
    :
        Density matrix tight-binding dictionary and Fermi energy.
    """
    ndim = len(list(h)[0])
    if ndim > 0:
        kham = tb_to_kgrid(h, nk=nk)
        _density_matrix_krid, fermi = density_matrix_kgrid(kham, filling)
        return (
            kgrid_to_tb(_density_matrix_krid),
            fermi,
        )
    else:
        _density_matrix, fermi = density_matrix_kgrid(h[()], filling)
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
