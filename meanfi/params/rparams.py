import numpy as np

from meanfi.params.param_transforms import (
    flat_to_tb,
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
    return tb_to_flat(tb)


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
    return flat_to_tb(tb_params, ndof, tb_keys)
