import itertools
import numpy as np

from pymf.tb.tb import _tb_type


def tb_to_khamvector(tb: _tb_type, nk: int) -> np.ndarray:
    """Evaluate a tight-binding dictionary on a k-space grid.

    Parameters
    ----------
    tb :
        Tight-binding dictionary to evaluate on a k-space grid.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        Tight-binding dictionary evaluated on a k-space grid.
        Has shape (nk, nk, ..., ndof, ndof), where ndof is number of internal degrees of freedom.
    """
    ndim = len(list(tb)[0])
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


def ifftn_to_tb(ifft_array: np.ndarray) -> _tb_type:
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
