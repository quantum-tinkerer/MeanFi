import numpy as np
from codes.tb.tb import add_tb


def density_matrix(kham, fermi_energy):
    """
     Parameters
     ----------
     kham : npndarray
         Hamiltonian in k-space of shape (len(dim), norbs, norbs)

    fermi_energy : float
         Fermi level

     Returns
     -------
     density_matrix_kgrid : np.ndarray
         Density matrix in k-space.

    Notes
    -----
    !! use filling instead of fermi_energy here?!

    """
    vals, vecs = np.linalg.eigh(kham)
    unocc_vals = vals > fermi_energy
    occ_vecs = vecs
    np.moveaxis(occ_vecs, -1, -2)[unocc_vals, :] = 0
    density_matrix_kgrid = occ_vecs @ np.moveaxis(occ_vecs, -1, -2).conj()
    return density_matrix_kgrid


def meanfield(density_matrix_tb, h_int, n=2):
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
