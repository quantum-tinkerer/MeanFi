import itertools
import numpy as np
from typing import Optional
from pymf.tb.tb import tb_type

ks_type = Optional[np.ndarray]


def tb_to_khamvector(tb: tb_type, nk: int, ks: ks_type = None) -> np.ndarray:
    """Real-space tight-binding model to hamiltonian on k-space grid.

    Parameters
    ----------
    tb :
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    nk :
        Number of k-points along each direction.
    ks :
        Set of k-points. Repeated for all directions.

    Returns
    -------
    :
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
    tb_array = tb_array.reshape([num_keys] + [1] * ndim + list(tb_array.shape[1:]))
    return np.sum(tb_array * k_dependency, axis=0)


def ifftn_to_tb(ifft_array: np.ndarray) -> tb_type:
    """Convert an array from ifftn to a tight-binding model format.

    Parameters
    ----------
    ifft_array :
        An array obtained from ifftn.

    Returns
    -------
    :
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    """
    size = ifft_array.shape[:-2]

    keys = [np.arange(-size[0] // 2 + 1, size[0] // 2) for i in range(len(size))]
    keys = itertools.product(*keys)
    return {tuple(k): ifft_array[tuple(k)] for k in keys}


def kham_to_tb(
    kham: np.ndarray,
    hopping_vecs: list[tuple[None] | tuple[int, ...]],
    ks: ks_type = None,
) -> tb_type:
    """Extract hopping matrices from Bloch Hamiltonian.

    Parameters
    ----------
    kham :
        Bloch Hamiltonian matrix kham[k_x, ..., k_n, i, j]
    hopping_vecs :
        List of hopping vectors, will be the keys to the tb.
    ks :
        Set of k-points. Repeated for all directions.
        If system is finite, this option is ignored.

    Returns
    -------
    :
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
