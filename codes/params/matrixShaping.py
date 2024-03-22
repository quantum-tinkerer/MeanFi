import numpy as np

def matrix_to_flat(matrix):
    """
    Flatten the upper triangle of a collection of matrices.

    Parameters:
    -----------
    matrix : nd-array
        Array with shape (..., n, n)
    """
    return matrix[..., *np.triu_indices(matrix.shape[-1])].flatten()

def flat_to_matrix(flat, shape):
    """
    Undo `matrix_to_flat`.
    
    Parameters:
    -----------
    flat : 1d-array
        Output from `matrix_to_flat`.
    shape : 1d-array
        Shape of the resulting tb model
    """
    matrix = np.zeros(shape, dtype=complex)
    matrix[..., *np.triu_indices(shape[-1])] = flat.reshape(*shape[:-2], -1)
    indices = np.arange(shape[-1])
    diagonal = matrix[..., indices, indices]
    matrix += np.moveaxis(matrix, -1, -2).conj()
    matrix[..., indices, indices] -= diagonal
    return matrix

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
    return z[:len(z)//2] + 1j * z[len(z)//2:]
