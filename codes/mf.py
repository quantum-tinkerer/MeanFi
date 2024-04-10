import numpy as np
from .tb.transforms import kfunc2tbFFT, kfunc2tbQuad, tb2kfunc
from .tb.tb import addTb
from scipy.integrate import nquad


def densityMatrixGenerator(hkfunc, E_F):
    """
    Generate a function that returns the density matrix at a given k-point.

    Parameters
    ----------
    hkfunc : function
        Function that return Hamiltonian at a given k-point.
    E_F : float
        Fermi level

    Returns
    -------
    densityMatrixFunc : function
        Returns a density matrix at a given k-point (kx, kx, ...)
    """

    def densityMatrixFunc(k):
        hk = hkfunc(k)
        vals, vecs = np.linalg.eigh(hk)
        unocc_vals = vals > E_F
        occ_vecs = vecs
        occ_vecs[:, unocc_vals] = 0

        # Outter products between eigenvectors
        return occ_vecs @ occ_vecs.T.conj()

    return densityMatrixFunc


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


def fermiOnGridkvector(kham, filling):

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


def fermiOnGrid(hkfunc, filling, nK=100, ndim=1):  # need to extend to 2D
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
    ks = np.linspace(-np.pi, np.pi, nK, endpoint=False)
    if ndim == 1:
        hkarray = np.array([hkfunc(k) for k in ks])
    if ndim == 2:
        hkarray = np.array([[hkfunc((kx, ky)) for kx in ks] for ky in ks])
    elif ndim > 2:
        raise NotImplementedError(
            "Fermi energy calculation is not implemented for ndim > 2"
        )

    vals = np.linalg.eigvalsh(hkarray)

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


# TODO: fix this function for N-dimensions
# def totalEnergy(rho, hk):
#     def integrand(k):
#         return np.real(np.trace(rho(k) @ hk(k)))
#     return quad(integrand, -np.pi, np.pi)[0]


def meanFieldFFTkvector(densityMatrixTb, h_int, n=2):

    localKey = tuple(np.zeros((n,), dtype=int))

    direct = {
        localKey: np.diag(
            np.einsum("pp,pn->n", densityMatrixTb[*localKey, :], h_int[localKey])
        )
    }
    exchange = {
        vec: -1 * h_int.get(vec, 0) * densityMatrixTb[*vec, :]
        for vec in frozenset(h_int)
    }
    return addTb(direct, exchange)


def meanFieldFFT(densityMatrix, int_model, n=2, nK=100):
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
                        np.einsum("pp,pn->n", densityMatrix[localKey], int_model[vec])
                    )
                    for vec in frozenset(int_model)
                ]
            ),
            axis=0,
        )
    }

    exchange = {
        vec: -1 * int_model.get(vec, 0) * densityMatrix[vec]  # / (2 * np.pi)#**2
        for vec in frozenset(int_model)
    }
    return addTb(direct, exchange)


def meanFieldQuad(densityMatrixFunc, h_int):  # TODO extent to 2D
    """
    Compute the mean-field in real space.

    Parameters
    ----------
    densityMatrixFunc : function
        Function that returns the density matrix at a given k-space vector.
    h_int : dict
        Interaction tb hopping dictionary.

    Returns
    -------
    dict
        Mean-field tb hopping dictionary.
    """
    from scipy.quad import quad_vec

    densityMatrixTb = kfunc2tbQuad(densityMatrixFunc)

    # Compute direct interaction
    intZero = tb2kfunc(h_int)(0)
    direct = quad_vec(lambda k: np.diag(densityMatrixFunc(k)), -np.pi, np.pi)[0]
    direct = np.diag(direct @ intZero) / (2 * np.pi)
    direct = {(0,): direct}
    exchange = {
        vec: -1 * (h_int.get(vec, 0) * densityMatrixTb(vec)) / (2 * np.pi)
        for vec in frozenset(h_int)
    }
    return addTb(direct, exchange)
