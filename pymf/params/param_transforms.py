import numpy as np

from pymf.tb.tb import tb_type


def tb_to_flat(tb: tb_type) -> np.ndarray:
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
    if len(list(tb)[0]) == 0:
        matrix = np.array(list(tb.values()))
        matrix = matrix.reshape((matrix.shape[-2], matrix.shape[-1]))
        return matrix[np.triu_indices(matrix.shape[-1])].flatten()
    N = len(tb.keys()) // 2 + 1
    sorted_vals = np.array(list(tb.values()))[np.lexsort(np.array(list(tb.keys())).T)]
    return sorted_vals[:N].flatten()


def flat_to_tb(
    tb_param_complex: np.ndarray,
    shape: tuple[int, int],
    tb_keys: list[tuple[None] | tuple[int, ...]],
) -> tb_type:
    """Reverse operation to `tb_to_flat`.

    It takes a flat complex 1d array and return the tight-binding dictionary.

    Parameters
    ----------
    tb_param_complex :
        1d complex array that parametrises the tb model.
    shape :
        Tuple (n, n) where n is the number of internal degrees of freedom
        (e.g. orbitals, spin, sublattice) within the tight-binding model.
    tb_keys :
        List of keys of the tight-binding dictionary.

    Returns
    -------
    tb :
        tight-binding dictionary
    """
    if len(tb_keys[0]) == 0:
        matrix = np.zeros((shape[-1], shape[-2]), dtype=complex)
        matrix[np.triu_indices(shape[-1])] = tb_param_complex
        matrix += matrix.conj().T
        matrix[np.diag_indices(shape[-1])] /= 2
        return {(): matrix}
    matrix = np.zeros(shape, dtype=complex)
    N = len(tb_keys) // 2 + 1
    matrix[:N] = tb_param_complex.reshape(N, *shape[1:])
    matrix[N:] = np.moveaxis(matrix[-(N + 1) :: -1], -1, -2).conj()

    tb_keys = np.array(list(tb_keys))
    sorted_keys = tb_keys[np.lexsort(tb_keys.T)]
    tb = dict(zip(map(tuple, sorted_keys), matrix))
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
