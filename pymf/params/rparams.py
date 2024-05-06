from pymf.params.param_transforms import (
    complex_to_real,
    flat_to_tb,
    real_to_complex,
    tb_to_flat,
)
import numpy as np
from pymf.tb.tb import tb_type


def tb_to_rparams(tb: tb_type) -> np.ndarray:
    """Convert a mean-field tight-binding model to a set of real parameters.

    Parameters
    ----------
    tb :
        Mean-field tight-binding model.

    Returns
    -------
    :
        1D real vector that parametrises the tb model.
    """
    return complex_to_real(tb_to_flat(tb))


def rparams_to_tb(
    r_params: np.ndarray, key_list: list[tuple[None] | tuple[int, ...]], size: int
) -> tb_type:
    """Extract mean-field tight-binding model from a set of real parameters.

    Parameters
    ----------
    r_params :
        Real parameters.
    key_list :
        List of the keys within the tight-binding model (all the hoppings).
    size :
        Number of internal degrees of freedom (e.g. orbitals, spin, sublattice) within
        the tight-binding model.

    Returns
    -------
    :
        Mean-field tight-binding model.
    """
    flat_matrix = real_to_complex(r_params)
    return flat_to_tb(flat_matrix, (len(key_list), size, size), key_list)
