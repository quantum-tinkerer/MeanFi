from codes.params.param_transforms import (
    complex_to_real,
    tb_to_flat,
    real_to_complex,
    flat_to_tb,
)


def tb_to_rparams(tb):
    """
    Convert a mean-field tight-binding model to a set of real parameters.

    Parameters
    ----------
    tb : dict
        Mean-field tight-binding model.

    Returns
    -------
    dict
        Real parameters.
    """
    return complex_to_real(tb_to_flat(tb))  # placeholder for now


def rparams_to_tb(rParams, key_list, size):
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
    return flat_to_tb(flat_matrix, (len(key_list), size, size), key_list)
