import itertools
import numpy as np
from typing import Optional
from pymf.tb.tb import tb_type

ks_type = Optional[np.ndarray]


def tb_to_khamvector(tb: tb_type, nk: int, ks: ks_type = None) -> np.ndarray:
    """Evaluate a tight-binding dictionary on a k-space grid.

    Parameters
    ----------
    tb :
        Tight-binding dictionary to evaluate on a k-space grid.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.
    ks :
        Set of points along which to evalaute the k-point grid. Repeated for all dimensions.
        If not provided, a linear grid is used based on nk.

    Returns
    -------
    :
        Tight-binding dictionary evaluated on a k-space grid.
        Has shape (nk, nk, ..., ndof, ndof), where ndof is number of internal degrees of freedom.
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
    """
    Convert the result of `scipy.fft.ifftn` to a tight-binding dictionary.

    Parameters
    ----------
    ifft_array :
        Result of `scipy.fft.ifftn` to convert to a tight-binding dictionary.
        The input to `scipy.fft.ifftn` should be from `tb_to_khamvector`.
    Returns
    -------
    :
        Tight-binding dictionary.
    """
    size = ifft_array.shape[:-2]

    keys = [np.arange(-size[0] // 2 + 1, size[0] // 2) for i in range(len(size))]
    keys = itertools.product(*keys)
    return {tuple(k): ifft_array[tuple(k)] for k in keys}


def kham_to_tb(
    kham: np.ndarray,
    tb_keys: list[tuple[None] | tuple[int, ...]],
    ks: ks_type = None,
) -> tb_type:
    """Convert a Hamiltonian evaluated on a k-grid to a tight-binding dictionary.

    Parameters
    ----------
    kham :
        Hamiltonian sampled on a grid of k-points with shape (nk, nk, ..., ndof, ndof),
        where ndof is number of internal degrees of freedom.
    tb_keys :
        List of keys for which to compute the tight-binding dictionary.
    ks :
        I have no clue why we need this, so this goes on the chopping board.
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
                np.exp(1j * np.einsum("ij,jk->ik", tb_keys, k_grid)),
                kham,
            )
            * (dk / (2 * np.pi)) ** ndim
        )

        h = {}
        for i, vector in enumerate(tb_keys):
            h[tuple(vector)] = hopps[i]

        return h
    else:
        return {(): kham}
