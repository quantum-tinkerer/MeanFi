import numpy as np

from itertools import chain
from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import sort_dict


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


def tb_to_params(tb: _tb_type) -> np.ndarray:
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


def params_to_tb(
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


def tb_to_projection(ham: _tb_type, basis_dict: dict) -> dict:
    """Project a tight-binding Hamiltonian onto an orthogonal basis and return the parameters.

    Parameters
    ----------
    ham: _tb_type
        A tight-binding `_tb_type` Hamiltonian.
    basis_dict: dict
        A `dict` with the orthogonal basis matrices in a list for every hopping.

    Returns
    -------
    A dictionary with the same hoppings as `ham` and `basis_dict` with an array of coefficients
      equal to the number of matrices in the basis per hopping.
    """
    coeffs = {
        hopping: np.array(
            [np.trace(basis.conj().T @ ham[hopping]) for basis in basis_dict[hopping]]
        )
        for hopping in ham
    }

    return coeffs


def projection_to_tb(coeffs: dict, basis_dict: dict) -> _tb_type:
    """Constructs a tight-binding Hamiltonian from a dictionary of coefficients and their corresponding basis matrices.
    The coefficients should be the scalar projection of the Hamiltonian onto the basis.

    Parameters
    ----------
    coeffs: dict
        A dictionary with an array of coefficients for every hopping.
    basis_dict: dict
        A dictionary with the same hoppings and an array of basis matrices for every hopping.

    Returns
    -------
    A tight-binding `_tb_type` Hamiltonian.
    """
    ham = {
        hopping: np.tensordot(coeff, basis_dict[hopping], 1)
        for hopping, coeff in coeffs.items()
    }

    return ham


def flatten_projection(coeffs: dict):
    """Flattens a coefficient dictionary into a list for passing to the optimizer."

    Parameters
    ----------
    coeffs: dict
        A dictionary with an array of coefficients for every hopping.

    Returns
    -------
    A flat list with the coefficients as values.
    """
    coeffs = sort_dict(coeffs)

    return np.array(list(chain.from_iterable(coeffs.values())))


def unflatten_projection(flat_coeffs: list, basis_dict: dict):
    """Construct a coefficient dictionary from a flattened list of coefficients using the dimensions of `basis_dict`.

    Parameters
    ----------
    flat_coeffs: list
        A flat list with the coefficients as values. This needs to have one coefficient for every matrix in `basis_dict`.
    basis_dict: dict
        A dictionary with hoppings and an array of basis matrices for every hopping.

    Returns
    -------
    A dictionary with an array of coefficients for every hopping.
    """
    coeffs = {}
    basis_dict = sort_dict(basis_dict)

    index_offset = 0
    for hopping in basis_dict.keys():
        n_coeffs = len(basis_dict[hopping])
        coeffs[hopping] = flat_coeffs[index_offset : index_offset + n_coeffs]

        index_offset += n_coeffs

    return coeffs
