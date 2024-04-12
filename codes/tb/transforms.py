import numpy as np
from scipy.fftpack import ifftn
import itertools as it


def tb2khamvector(h_0, nK, ndim):
    """
    Real-space tight-binding model to hamiltonian on k-space grid.

    Parameters
    ----------
    h_0 : dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    nK : int
        Number of k-points along each direction.
    ndim : int
        Number of dimensions.

    Returns
    -------
    ndarray
        Hamiltonian evaluated on a k-point grid.

    """

    ks = np.linspace(-np.pi, np.pi, nK, endpoint=False)
    ks = np.concatenate((ks[nK // 2 :], ks[: nK // 2]), axis=0)  # shift for ifft
    kgrid = np.meshgrid(ks, ks, indexing="ij")

    num_keys = len(list(h_0.keys()))
    tb_array = np.array(list(h_0.values()))
    keys = np.array(list(h_0.keys()))

    k_dependency = np.exp(-1j * np.tensordot(keys, kgrid, 1))[
        (...,) + (np.newaxis,) * 2
    ]
    tb_array = tb_array.reshape(
        np.concatenate(([num_keys], [1] * ndim, tb_array.shape[1:]))
    )
    return np.sum(tb_array * k_dependency, axis=0)


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


def ifftn2tb(ifftArray):
    """
    Converts an array from ifftn to a tight-binding model format.

    Parameters
    ----------
    ifftArray : ndarray
        An array obtained from ifftn.

    Returns
    -------
    dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    """

    size = ifftArray.shape[:-2]

    keys = [np.arange(-size[0] // 2 + 1, size[0] // 2) for i in range(len(size))]
    keys = it.product(*keys)
    return {tuple(k): ifftArray[tuple(k)] for k in keys}


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
    for k in it.product(*k_pts):
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


def kfunc2tb(kfunc, nSamples, ndim=1):
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

    dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
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
    ifftnArray = ifftn(kfuncOnGrid, axes=np.arange(ndim))
    return ifftn2tb(ifftnArray)


def hk2h_0(hk, hopping_vecs, ks=None):
    """
    Extract hopping matrices from Bloch Hamiltonian.

    Parameters:
    -----------
    hk : nd-array
        Bloch Hamiltonian matrix hk[k_x, ..., k_n, i, j]
    h_0 : dict
        Tight-binding model of non-interacting systems.
    h_int : dict
        Tight-binding model for interacting Hamiltonian.
    ks : 1D-array
        Set of k-points. Repeated for all directions. If the system is finite, `ks=None`.

    Returns:
    --------
    scf_model : dict
        TIght-binding model of Hartree-Fock solution.
    """
    if ks is not None:
        ndim = len(hk.shape) - 2
        dk = np.diff(ks)[0]
        nk = len(ks)
        k_pts = np.tile(ks, ndim).reshape(ndim, nk)
        k_grid = np.array(np.meshgrid(*k_pts))
        k_grid = k_grid.reshape(k_grid.shape[0], np.prod(k_grid.shape[1:]))
        hk = hk.reshape(np.prod(hk.shape[:ndim]), *hk.shape[-2:])

        hopps = (
            np.einsum(
                "ij,jkl->ikl",
                np.exp(1j * np.einsum("ij,jk->ik", hopping_vecs, k_grid)),
                hk,
            )
            * (dk / (2 * np.pi)) ** ndim
        )

        h_0 = {}
        for i, vector in enumerate(hopping_vecs):
            h_0[tuple(vector)] = hopps[i]

        return h_0
    else:
        return {(): hk}
