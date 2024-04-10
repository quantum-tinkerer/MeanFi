import numpy as np
from .utils import quad_vecNDim
from scipy.fftpack import ifftn
from itertools import product


def tb2kfunc(h_0):
    """
    Fourier transforms a real-space tight-binding model to a k-space function.

    Parameters
    ----------
    h_0 : dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.

    Returns
    -------
    function
        A function that takes a k-space vector and returns a complex np.array.
    """

    def bloch_ham(k):
        ham = 0
        for vector in h_0.keys():
            ham += h_0[vector] * np.exp(-1j * np.dot(k, np.array(vector, dtype=float)))
        return ham

    return bloch_ham


def kfunc2kham(nk, hk, dim, return_ks=False, hermitian=True):
    """
    Evaluates Hamiltonian on a k-point grid.

    Paramters:
    ----------
    nk : int
        Number of k-points along each direction.
    hk : function
        Calculates the Hamiltonian at a given k-point.
    return_ks : bool
        If `True`, returns k-points.

    Returns:
    --------
    ham : nd.array
        Hamiltonian evaluated on a k-point grid from k-points
        along each direction evaluated from zero to 2*pi.
        The indices are ordered as [k_1, ... , k_n, i, j], where
        `k_m` corresponding to the k-point element along each
        direction and `i` and `j` are the internal degrees of freedom.
    ks : 1D-array
        List of k-points over all directions. Only returned if `return_ks=True`.
    """
    ks = np.linspace(
        -np.pi, np.pi, nk, endpoint=False
    )  # for now nk need to be even such that 0 is in the middle
    ks = np.concatenate((ks[nk // 2 :], ks[: nk // 2]), axis=0)  # shift for ifft

    k_pts = np.tile(ks, dim).reshape(dim, nk)

    ham = []
    for k in product(*k_pts):
        ham.append(hk(k))
    ham = np.array(ham)
    if hermitian:
        assert np.allclose(
            ham, np.transpose(ham, (0, 2, 1)).conj()
        ), "Tight-binding provided is non-Hermitian. Not supported yet"
    shape = (*[nk] * dim, ham.shape[-1], ham.shape[-1])
    if return_ks:
        return ham.reshape(*shape), ks
    else:
        return ham.reshape(*shape)


def tb2kham(h_0, nK=200, ndim=1):
    kfunc = tb2kfunc(h_0)
    kham = kfunc2kham(nK, kfunc, ndim)
    return kham


def kfunc2tbFFT(kfunc, nSamples, ndim=1):
    """
    Applies FFT on a k-space function to obtain a real-space components.

    Parameters
    ----------
    kfunc : function
        A function that takes a k-space vector and returns a np.array.
    nSamples : int
        Number of samples to take in k-space.

    Returns
    -------

    ndarray
        An array with real-space components of kfuncs
    """

    ks = np.linspace(
        -np.pi, np.pi, nSamples, endpoint=False
    )  # for now nSamples need to be even such that 0 is in the middle
    ks = np.concatenate(
        (ks[nSamples // 2 :], ks[: nSamples // 2]), axis=0
    )  # shift for ifft

    if ndim == 1:
        kfuncOnGrid = np.array([kfunc(k) for k in ks])
    if ndim == 2:
        kfuncOnGrid = np.array([[kfunc((kx, ky)) for ky in ks] for kx in ks])
    if ndim > 2:
        raise NotImplementedError("n > 2 not implemented")
    return ifftn(kfuncOnGrid, axes=np.arange(ndim))


def kdens2tbFFT(kdens, ndim=1):

    return ifftn(kdens, axes=np.arange(ndim))


def kfunc2tbQuad(kfunc, ndim=1):
    """
    Inverse Fourier transforms a k-space function to a real-space function using a
    ndim quadrature integration.

    Parameters
    ----------
    kfunc : function
        A function that takes a k-space vector and returns a np.array.
    ndim : int
        Dimension of the k-space

    Returns
    -------
    function
        A function that takes a real-space integer vector and returns a np.array.
    """

    def tbfunc(vector):
        def integrand(k):
            return (
                kfunc(k)
                * np.exp(1j * np.dot(k, np.array(vector, dtype=float)))
                / (2 * np.pi)
            )

        return quad_vecNDim(integrand, -np.pi, np.pi, ndim=ndim)[0]

    return tbfunc
