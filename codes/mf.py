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
        raise NotImplementedError("Fermi energy calculation is not implemented for ndim > 2")

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


def totalEnergy(rho, hk, ndim=1, quad_kwargs={}):
    """
    Compute the total energy of a system.

    Parameters
    ----------
    rho : function
        Function that returns the density matrix at a given k-point.
    hk : function
        Function that returns the Hamiltonian at a given k-point.
    ndim : int  
        Dimension of the system.
    quad_kwargs : dict
        Keyword arguments for the nquad function.

    Returns
    -------
    float
        Total energy.
    """
    def integrand(*k):
        return np.real(np.trace(rho(k) @ hk(k)))
    bounds = [(-np.pi, np.pi) for _ in range(ndim)]
    return nquad(integrand, bounds, **quad_kwargs)[0]

def meanFieldFFT(densityMatrixFunc, int_model, n=2, nK=100):
    """
    Compute the mean-field in k-space.

    Parameters
    ----------
    rhofunc : function
        Function that returns the density matrix at a given k-space vector.
    int_model : dict
        Interaction tb model.

    Returns
    -------
    dict
        Mean-field tb model.
    """
    densityMatrixTb = kfunc2tbFFT(densityMatrixFunc, nK, n)
    localKey = tuple(np.zeros((n,), dtype=int))

    direct = {
        localKey: np.sum(np.array([np.diag(
            np.einsum("pp,pn->n", densityMatrixTb[*localKey, :], int_model[vec]))
            for vec in frozenset(int_model)]), axis=0
        )
    }

    exchange = {
        vec: -1 * int_model.get(vec, 0) * densityMatrixTb[*vec, :] #/ (2 * np.pi)#**2
        for vec in frozenset(int_model)
    }
    return addTb(direct, exchange)


def meanFieldQuad(densityMatrixFunc, int_model):  # TODO extent to 2D
    """
    Compute the mean-field in real space.

    Parameters
    ----------
    densityMatrixFunc : function
        Function that returns the density matrix at a given k-space vector.
    int_model : dict
        Interaction tb model.

    Returns
    -------
    dict
        Mean-field tb model.
    """
    from scipy.quad import quad_vec

    densityMatrixTb = kfunc2tbQuad(densityMatrixFunc)

    # Compute direct interaction
    intZero = tb2kfunc(int_model)(0)
    direct = quad_vec(lambda k: np.diag(densityMatrixFunc(k)), -np.pi, np.pi)[0]
    direct = np.diag(direct @ intZero) / (2 * np.pi)
    direct = {(0,): direct}
    exchange = {
        vec: -1 * (int_model.get(vec, 0) * densityMatrixTb(vec)) / (2 * np.pi)
        for vec in frozenset(int_model)
    }
    return addTb(direct, exchange)
