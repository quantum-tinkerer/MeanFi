import itertools
import numpy as np
from typing import Callable
from scipy.fftpack import ifftn
from qsymm import bloch_family

from collections import defaultdict
from tb.tb import _tb_type


def tb_to_kgrid(tb: _tb_type, nk: int) -> np.ndarray:
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


def kgrid_to_tb(kgrid_array: np.ndarray) -> _tb_type:
    """
    Convert a k-space grid array to a tight-binding dictionary.

    Parameters
    ----------
    kgrid_array :
        K-space grid array to convert to a tight-binding dictionary.
        The array should be of shape (nk, nk, ..., ndof, ndof),
        where ndof is number of internal degrees of freedom.
    Returns
    -------
    :
        Tight-binding dictionary.
    """
    ndim = len(kgrid_array.shape) - 2
    return ifftn_to_tb(ifftn(kgrid_array, axes=np.arange(ndim)))


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


def tb_to_kfunc(tb: _tb_type) -> Callable:
    """
    Fourier transforms a real-space tight-binding model to a k-space function.

    Parameters
    ----------
    tb :
        Tight-binding dictionary.

    Returns
    -------
    :
        A function that takes a k-space vector and returns a complex np.array.

    Notes
    -----
    Function doesn't work for zero dimensions.
    """

    def kfunc(k):
        ham = 0
        for vector in tb.keys():
            ham += tb[vector] * np.exp(-1j * np.dot(k, np.array(vector, dtype=float)))
        return ham

    return kfunc


def tb_to_ham_fam(hams: tuple, symmetries: list, nsites: int = 0) -> list:
    """Generate a Hamiltonian Family for a given tight-binding dictionary.

    Parameters
    ----------
    hams: tuple
        A tuple of `_tb_type` Hamiltonians.
    symmteries: list
        A list of symmetries from `qsymm`.
    nsites: int
        Number of sites in a unit cell. Currently not used.

    Returns
    -------
    A basis of all allowed Hamiltonians for the chosen symmetries and hoppings in the form of a `qsymm` list of `BlochModel`s.
    """

    hopp_real = list(set().union(*hams) if len(hams) > 1 else hams[0].keys())

    ndof = hams[0][next(iter(hams[0]))].shape[-1]

    hoppings = []
    for vec in hopp_real:
        hoppings.append(("a", "a", vec))

    nsites = 0
    norbs = [("a", ndof - nsites)]

    ham_fam = bloch_family(hoppings, symmetries, norbs, bloch_model=True)

    return ham_fam


def qsymm_key_to_tb(ham_fam_key: tuple) -> tuple:
    """Converts a `qsymm` `BlochModel` hopping to a `_tb_type` hopping.

    Parameters
    ----------
    ham_fam_key: tuple
        A `qsymm` `BlochModel` style hopping.

    Returns
    -------
    A `_tb_type` hopping.
    """
    hopping, site = ham_fam_key
    return tuple(hopping.astype(int))


def ham_fam_to_tb_dict(ham_fam: list) -> dict:
    """Converts a Hamiltonian Family into a dict with a list of allowed matrices per hopping.
    Translation layer between a `bloch_family` and `_tb_type`.

    Parameters
    ----------
    ham_fam: list
        A list of `qsymm` `BlochModels`.

    Returns
    -------
    A `dict` with the appropriate basis matrices in a list for every hopping.
    """
    ham_fam_dict = defaultdict(list)
    for bloch in ham_fam:
        for hop, matrix in bloch.items():
            key = qsymm_key_to_tb(hop)
            ham_fam_dict[key].append(matrix)

    return ham_fam_dict


def ham_fam_to_ort_basis(ham_fam: list) -> dict:
    """Finds an orthogonal `_tb_type` basis for a family of Hamiltonians using QR decomposition.

    Parameters
    ----------
    ham_fam: list
        A list of `qsymm` `BlochModels`

    Returns
    -------
    A `dict` with the orthogonal basis matrices in a list for every hopping.
    """
    ham_fam_dict = ham_fam_to_tb_dict(ham_fam)

    ort_dict = {}
    for hopping in ham_fam_dict:
        if hopping not in ham_fam_dict:
            raise KeyError(f"No basis found for hopping {hopping}.")

        basis = ham_fam_dict[hopping]

        A = np.column_stack([H.flatten() for H in basis])
        Q = np.linalg.qr(A, mode="reduced")[0]

        ort_dict[hopping] = np.reshape(Q, (len(basis),) + basis[0].shape)

    return ort_dict
