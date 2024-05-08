import numpy as np

from meanfi.params.param_transforms import (
    complex_to_real,
    flat_to_tb,
    real_to_complex,
    tb_to_flat,
)
from meanfi.tb.tb import _tb_type


def tb_to_rparams(tb: _tb_type) -> np.ndarray:
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
    return complex_to_real(tb_to_flat(tb))


def rparams_to_tb(
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
    flat_matrix = real_to_complex(tb_params)
    return flat_to_tb(flat_matrix, ndof, tb_keys)
