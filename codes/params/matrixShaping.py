import numpy as np

def hop_dict_to_flat(hop_dict):
    """
    Convert a hermitian tight-binding dictionary to flat complex matrix.

    Parameters:
    -----------
    hop_dict : dict with nd-array elements
        Hermitian tigh-binding dictionary
    
    Returns:
    -----------
    flat : complex 1d numpy array
        Flattened tight-binding dictionary
    """
    sorted_vals = np.array(list(hop_dict.values()))[
        np.lexsort(np.array(list(hop_dict)).T)
    ]
    flat = sorted_vals[..., *np.triu_indices(sorted_vals.shape[-1])].flatten()
    return flat

def flat_to_hop_dict(flat, shape, hop_dict_keys):
    """
    Reverse operation to `hop_dict_to_flat` that takes a flat complex 1d array
    and return the tight-binding dictionary.

    Parameters:
    -----------
    flat : dict with nd-array elements
        Hermitian tigh-binding dictionary
    shape : tuple
        shape of the hop_dict elements
    hop_dict_keys : iterable
        original hop_dict key elements
    
    Returns:
    -----------
    hop_dict : dict
        tight-binding dictionary
    """
    matrix = np.zeros(shape, dtype=complex)
    matrix[..., *np.triu_indices(shape[-1])] = flat.reshape(*shape[:-2], -1)
    indices = np.arange(shape[-1])
    diagonal = matrix[..., indices, indices]
    matrix += np.moveaxis(matrix[-1::-1], -1, -2).conj()
    matrix[..., indices, indices] -= diagonal

    hop_dict_keys = np.array(list(hop_dict_keys))
    sorted_keys = hop_dict_keys[np.lexsort(hop_dict_keys.T)]
    hop_dict = dict(zip(map(tuple, sorted_keys), matrix))
    return hop_dict

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