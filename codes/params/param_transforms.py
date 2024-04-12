import numpy as np


def tb_to_flat(tb):
    """
    Convert a hermitian tight-binding dictionary to flat complex matrix.

    Parameters:
    -----------
    tb : dict with nd-array elements
        Hermitian tigh-binding dictionary

    Returns:
    -----------
    flat : complex 1d numpy array
        Flattened tight-binding dictionary
    """
    N = len(tb.keys()) // 2 + 1
    sorted_vals = np.array(list(tb.values()))[np.lexsort(np.array(list(tb.keys())).T)]
    return sorted_vals[:N].flatten()


def flat_to_tb(flat, shape, tb_keys):
    """
    Reverse operation to `tb_to_flat` that takes a flat complex 1d array
    and return the tight-binding dictionary.

    Parameters:
    -----------
    flat : dict with nd-array elements
        Hermitian tigh-binding dictionary
    shape : tuple
        shape of the tb elements
    tb_keys : iterable
        original tb key elements

    Returns:
    -----------
    tb : dict
        tight-binding dictionary
    """
    matrix = np.zeros(shape, dtype=complex)
    N = len(tb_keys) // 2 + 1
    matrix[:N] = flat.reshape(N, *shape[1:])
    matrix[N:] = np.moveaxis(matrix[-(N + 1) :: -1], -1, -2).conj()

    tb_keys = np.array(list(tb_keys))
    sorted_keys = tb_keys[np.lexsort(tb_keys.T)]
    tb = dict(zip(map(tuple, sorted_keys), matrix))
    return tb


def complex_to_real(z):
    """
    Split real and imaginary parts of a complex array.

    Parameters:
    -----------
    z : array
    """
    return np.concatenate((np.real(z), np.imag(z)))


def real_to_complex(z):
    """
    Undo `complex_to_real`.
    """
    return z[: len(z) // 2] + 1j * z[len(z) // 2 :]
