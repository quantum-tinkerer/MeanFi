import itertools as it

import numpy as np


def tb_to_khamvector(tb, nk, ks=None):
    """Real-space tight-binding model to hamiltonian on k-space grid.

    Parameters
    ----------
    tb : dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    nk : int
        Number of k-points along each direction.
    ks : 1D-array
        Set of k-points. Repeated for all directions.

    Returns
    -------
    ndarray
        Hamiltonian evaluated on a k-point grid.

    """
    ndim = len(list(tb)[0])
    if ks is None:
        ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
        ks = np.concatenate((ks[nk // 2 :], ks[: nk // 2]), axis=0)  # shift for ifft
    kgrid = np.meshgrid(*([ks] * ndim), indexing="ij")

    num_keys = len(list(tb.keys()))
    tb_array = np.array(list(tb.values()))
    keys = np.array(list(tb.keys()))

    k_dependency = np.exp(-1j * np.tensordot(keys, kgrid, 1))[
        (...,) + (np.newaxis,) * 2
    ]
    tb_array = tb_array.reshape(
        np.concatenate(([num_keys], [1] * ndim, tb_array.shape[1:])).astype(int)
    )
    return np.sum(tb_array * k_dependency, axis=0)


def ifftn_to_tb(ifft_array):
    """Convert an array from ifftn to a tight-binding model format.

    Parameters
    ----------
    ifft_array : ndarray
        An array obtained from ifftn.

    Returns
    -------
    dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    """
    size = ifft_array.shape[:-2]

    keys = [np.arange(-size[0] // 2 + 1, size[0] // 2) for i in range(len(size))]
    keys = it.product(*keys)
    return {tuple(k): ifft_array[tuple(k)] for k in keys}


def kham_to_tb(kham, hopping_vecs, ks=None):
    """Extract hopping matrices from Bloch Hamiltonian.

    Parameters
    ----------
    kham : nd-array
        Bloch Hamiltonian matrix kham[k_x, ..., k_n, i, j]
    hopping_vecs : list
        List of hopping vectors, will be the keys to the tb.
    ks : 1D-array
        Set of k-points. Repeated for all directions. If the system is finite,
        ks=None`.

    Returns
    -------
    scf_model : dict
        Tight-binding model of Hartree-Fock solution.
    """
    if ks is not None:
        ndim = len(kham.shape) - 2
        dk = np.diff(ks)[0]
        nk = len(ks)
        k_pts = np.tile(ks, ndim).reshape(ndim, nk)
        k_grid = np.array(np.meshgrid(*k_pts))
        k_grid = k_grid.reshape(k_grid.shape[0], np.prod(k_grid.shape[1:]))
        kham = kham.reshape(np.prod(kham.shape[:ndim]), *kham.shape[-2:])

        hopps = (
            np.einsum(
                "ij,jkl->ikl",
                np.exp(1j * np.einsum("ij,jk->ik", hopping_vecs, k_grid)),
                kham,
            )
            * (dk / (2 * np.pi)) ** ndim
        )

        h_0 = {}
        for i, vector in enumerate(hopping_vecs):
            h_0[tuple(vector)] = hopps[i]

        return h_0
    else:
        return {(): kham}
