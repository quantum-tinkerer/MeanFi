from codes.params.matrixShaping import (
    complex_to_real,
    hop_dict_to_flat,
    real_to_complex,
    flat_to_hop_dict,
)


def mf_to_rparams(mf_model):
    """
    Convert a mean-field tight-binding model to a set of real parameters.

    Parameters
    ----------
    mf_model : dict
        Mean-field tight-binding model.

    Returns
    -------
    dict
        Real parameters.
    """
    return complex_to_real(hop_dict_to_flat(mf_model))  # placeholder for now


def rparams_to_mf(rParams, key_list, size):
    """
    Extract mean-field tight-binding model from a set of real parameters.

    Parameters
    ----------
    r_params : dict
        Real parameters.
    shape : tuple
        Shape of the mean-field tight-binding model.

    Returns
    -------
    dict
        Mean-field tight-binding model.
    """

    flat_matrix = real_to_complex(rParams)
    return flat_to_hop_dict(flat_matrix, (len(key_list), size, size), key_list)
