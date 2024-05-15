import numpy as np

from meanfi.tb.tb import _tb_type


def tb_to_flat(tb: _tb_type) -> np.ndarray:
    """Parametrise a hermitian tight-binding dictionary by a flat complex vector.

    Parameters
    ----------
    tb :
        Hermitian tigh-binding dictionary

    Returns
    -------
    :
        1D complex array that parametrises the tight-binding dictionary.
    """
    ndim = len(list(tb)[0])
    onsite_key = tuple(np.zeros((ndim,), dtype=int))

    hopping_keys = sorted([key for key in tb.keys() if key != onsite_key])
    hopping_keys = hopping_keys[: len(hopping_keys) // 2]

    onsite_val = tb[onsite_key][np.triu_indices(tb[onsite_key].shape[-1])].flatten()
    hopping_vals = [tb[key].flatten() for key in hopping_keys]
    return np.concatenate([onsite_val] + hopping_vals)


def flat_to_tb(
    tb_param_complex: np.ndarray,
    ndof: int,
    tb_keys: list[tuple[None] | tuple[int, ...]],
) -> _tb_type:
    """Reverse operation to `tb_to_flat`.

    It takes a flat complex 1d array and return the tight-binding dictionary.

    Parameters
    ----------
    tb_param_complex :
        1d complex array that parametrises the tb model.
    ndof :
        Number internal degrees of freedom within the unit cell.
    tb_keys :
        List of keys of the tight-binding dictionary.

    Returns
    -------
    tb :
        tight-binding dictionary
    """
    ndim = len(tb_keys[0])

    hopping_shape = (len(tb_keys) - 1, ndof, ndof)

    onsite_idxs = ndof + ndof * (ndof - 1) // 2
    onsite_key = tuple(np.zeros((ndim,), dtype=int))

    # first build onsite matrix
    onsite_matrix = np.zeros((ndof, ndof), dtype=complex)
    onsite_matrix[np.triu_indices(ndof)] = tb_param_complex[:onsite_idxs]
    onsite_matrix += onsite_matrix.conj().T
    onsite_matrix[np.diag_indices(ndof)] /= 2

    # then build hopping matrices
    hopping_matrices = np.zeros(hopping_shape, dtype=complex)
    N = len(tb_keys) // 2
    hopping_matrices[:N] = tb_param_complex[onsite_idxs:].reshape(N, *hopping_shape[1:])
    hopping_matrices[N:] = np.moveaxis(
        np.flip(hopping_matrices[:N], axis=0), -1, -2
    ).conj()

    # combine all into a dictionary
    hopping_keys = sorted([key for key in tb_keys if key != onsite_key])
    tb = {key: hopping_matrices[i] for i, key in enumerate(hopping_keys)}
    tb[onsite_key] = onsite_matrix
    return tb


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
