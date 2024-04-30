import numpy as np
from codes.tb.tb import add_tb
from codes.tb.transforms import tb_to_khamvector, ifftn_to_tb
from scipy.fftpack import ifftn


def density_matrix_kgrid(kham, fermi_energy):
    """
     Parameters
     ----------
     kham : npndarray
         Hamiltonian in k-space of shape (len(dim), norbs, norbs)
     fermi_energy : float
         Fermi level

     Returns
     -------
     np.ndarray
         Density matrix in k-space.

    Notes
    -----
    !! use filling instead of fermi_energy here?!

    """
    vals, vecs = np.linalg.eigh(kham)
    unocc_vals = vals > fermi_energy
    occ_vecs = vecs
    np.moveaxis(occ_vecs, -1, -2)[unocc_vals, :] = 0
    rho_krid = occ_vecs @ np.moveaxis(occ_vecs, -1, -2).conj()
    return rho_krid

def density_matrix(h, filling, nk):
    """
    Compute the density matrix in real-space tight-binding format.

    Parameters
    ----------
    h : dict
        Tight-binding model.
    filling : float
        Filling of the system.
    nk : int
        Number of k-points in the grid.

    Returns
    -------
    (dict, float)
        Density matrix in real-space tight-binding format and Fermi energy.
    """
    ndim = len(list(h)[0])
    if ndim > 0:
        kham = tb_to_khamvector(h, nk=nk)
        fermi = fermi_on_grid(kham, filling)
        return (
            ifftn_to_tb(ifftn(density_matrix_kgrid(kham, fermi), axes=np.arange(ndim))),
            fermi,
        )
    else:
        fermi = fermi_on_grid(h[()], filling)
        return {(): density_matrix_kgrid(h[()], fermi)}

def meanfield(density_matrix_tb, h_int):
    """
    Compute the mean-field in k-space.

    Parameters
    ----------
    density_matrix : dict
        Density matrix in real-space tight-binding format.
    int_model : dict
        Interaction tb model.

    Returns
    -------
    dict
        Mean-field tb model.
    """
    n = len(list(density_matrix_tb)[0])
    local_key = tuple(np.zeros((n,), dtype=int))

    direct = {
        local_key: np.sum(
            np.array(
                [
                    np.diag(
                        np.einsum("pp,pn->n", density_matrix_tb[local_key], h_int[vec])
                    )
                    for vec in frozenset(h_int)
                ]
            ),
            axis=0,
        )
    }

    exchange = {
        vec: -1 * h_int.get(vec, 0) * density_matrix_tb[vec]  # / (2 * np.pi)#**2
        for vec in frozenset(h_int)
    }
    return add_tb(direct, exchange)


def fermi_on_grid(kham, filling):
    """
     Compute the Fermi energy on a grid of k-points.

     Parameters
     ----------
     hkfunc : function
         Function that returns the Hamiltonian at a given k-point.
     nk : int
         Number of k-points in the grid.
     Returns
     -------
    fermi_energy : float
         Fermi energy
    """

    vals = np.linalg.eigvalsh(kham)

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
