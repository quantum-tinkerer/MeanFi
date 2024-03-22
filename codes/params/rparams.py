import numpy as np
from .matrixShaping import (complex_to_real, matrix_to_flat,
                             real_to_complex, flat_to_matrix)

def mf2rParams(mf_model):
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
    return complex_to_real(matrix_to_flat(np.array([mf_model[key] for key in mf_model]))) # placeholder for now

def rParams2mf(rParams, keyList, size):
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

    flatMatrix = real_to_complex(rParams)
    shapedMatrix = flat_to_matrix(flatMatrix, (len(keyList), size, size))
    return {keyList[i] : shapedMatrix[i, :] for i  in range(len(keyList))}

    ## TODO: test the function via performing to and back conversion on a random model
def compareDicts(dict1, dict2):
    for key in dict1.keys():
        assert np.allclose(dict1[key], dict2[key])