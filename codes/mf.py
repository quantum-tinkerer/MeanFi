import numpy as np
from codes.tb.tb import addTb


def densityMatrix(kham, E_F):
    """
    Parameters
    ----------
    kham : npndarray
        Hamiltonian in k-space of shape (len(dim), norbs, norbs)

    E_F : float
        Fermi level

    Returns
    -------
    densityMatrixKgrid : np.ndarray
        Density matrix in k-space.

    """
    vals, vecs = np.linalg.eigh(kham)
    unocc_vals = vals > E_F
    occ_vecs = vecs
    np.moveaxis(occ_vecs, -1, 2)[unocc_vals, :] = 0
    densityMatrixKgrid = occ_vecs @ np.moveaxis(occ_vecs, -1, -2).conj()
    return densityMatrixKgrid


def fermiOnGrid(kham, filling):
    """
    Compute the Fermi energy on a grid of k-points.

    Parameters
    ----------
    hkfunc : function
        Function that returns the Hamiltonian at a given k-point.
    Nk : int
        Number of k-points in the grid.
    Returns
    -------
    E_F : float
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


def meanField(densityMatrixTb, h_int, n=2):
    """
    Compute the mean-field in k-space.

    Parameters
    ----------
    densityMatrix : dict
        Density matrix in real-space tight-binding format.
    int_model : dict
        Interaction tb model.

    Returns
    -------
    dict
        Mean-field tb model.
    """

    localKey = tuple(np.zeros((n,), dtype=int))

    direct = {
        localKey: np.sum(
            np.array(
                [
                    np.diag(
                        np.einsum("pp,pn->n", densityMatrixTb[localKey], h_int[vec])
                    )
                    for vec in frozenset(h_int)
                ]
            ),
            axis=0,
        )
    }

    exchange = {
        vec: -1 * h_int.get(vec, 0) * densityMatrixTb[vec]  # / (2 * np.pi)#**2
        for vec in frozenset(h_int)
    }
    return addTb(direct, exchange)
