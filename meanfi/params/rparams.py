import numpy as np

from meanfi.tb.tb import _tb_type


def complex_to_real(z: np.ndarray) -> np.ndarray:
    """Split and concatenate real and imaginary parts of a complex array.

    Parameters
    ----------
    z :
        Complex array.

    Returns
    -------
    :
        Real array that concatenates the real and imaginary parts of the input array.
    """
    return np.concatenate((np.real(z), np.imag(z)))


def real_to_complex(z: np.ndarray) -> np.ndarray:
    """Undo `complex_to_real`."""
    return z[: len(z) // 2] + 1j * z[len(z) // 2 :]


def tb_to_rparams(tb: _tb_type) -> np.ndarray:
    """Parametrise a hermitian tight-binding dictionary by a real vector.

    Parameters
    ----------
    tb :
        tight-binding dictionary.

    Returns
    -------
    :
        1D real vector that parametrises the tight-binding dictionary.
    """
    ndim = len(list(tb)[0])
    onsite_key = tuple(np.zeros((ndim,), dtype=int))

    hopping_keys = sorted([key for key in tb.keys() if key != onsite_key])
    hopping_keys = hopping_keys[: len(hopping_keys) // 2]

    onsite_upper_vals = tb[onsite_key][
        np.triu_indices(tb[onsite_key].shape[-1], k=1)
    ].flatten()
    onsite_diag_vals = np.diag(tb[onsite_key]).real.flatten()

    hopping_vals = [tb[key].flatten() for key in hopping_keys]
    complex_vals = np.concatenate([onsite_upper_vals, *hopping_vals])
    return np.concatenate([onsite_diag_vals, complex_to_real(complex_vals)])


def rparams_to_tb(
    tb_params: np.ndarray, tb_keys: list[tuple[None] | tuple[int, ...]], ndof: int
) -> _tb_type:
    """Extract a hermitian tight-binding dictionary from a real vector parametrisation.

    Parameters
    ----------
    tb_params :
        1D real array that parametrises the tight-binding dictionary.
    tb_keys :
        List of keys of the tight-binding dictionary.
    ndof :
        Number internal degrees of freedom within the unit cell.

    Returns
    -------
    :
        Tight-biding dictionary.
    """
    ndim = len(tb_keys[0])

    hopping_shape = (len(tb_keys) - 1, ndof, ndof)

    onsite_idxs = ndof + ndof * (ndof - 1) // 2
    onsite_key = tuple(np.zeros((ndim,), dtype=int))

    # reconstruct the complex values
    onsite_diag_vals = tb_params[:ndof]
    complex_vals = real_to_complex(tb_params[ndof:])
    onsite_upper_vals = complex_vals[: onsite_idxs - ndof]
    hopping_vals = complex_vals[(onsite_idxs - ndof) :]

    # first build onsite matrix
    onsite_matrix = np.zeros((ndof, ndof), dtype=complex)
    onsite_matrix[np.triu_indices(ndof, k=1)] = onsite_upper_vals
    onsite_matrix += onsite_matrix.conj().T
    onsite_matrix[np.diag_indices(ndof)] = onsite_diag_vals

    # then build hopping matrices
    hopping_matrices = np.zeros(hopping_shape, dtype=complex)
    N = len(tb_keys) // 2
    hopping_matrices[:N] = hopping_vals.reshape(N, *hopping_shape[1:])
    hopping_matrices[N:] = np.moveaxis(
        np.flip(hopping_matrices[:N], axis=0), -1, -2
    ).conj()

    # combine all into a dictionary
    hopping_keys = sorted([key for key in tb_keys if key != onsite_key])
    tb = {key: hopping_matrices[i] for i, key in enumerate(hopping_keys)}
    tb[onsite_key] = onsite_matrix
    return tb
