import numpy as np

tb_type = dict[tuple[None] | tuple[int, ...], np.ndarray]


def add_tb(tb1: tb_type, tb2: tb_type) -> tb_type:
    """Add up two tight-binding models together.

    Parameters
    ----------
    tb1 :
        Tight-binding model.
    tb2 :
        Tight-binding model.

    Returns
    -------
    :
        Sum of the two tight-binding models.
    """
    return {k: tb1.get(k, 0) + tb2.get(k, 0) for k in frozenset(tb1) | frozenset(tb2)}


def scale_tb(tb: tb_type, scale: float) -> tb_type:
    """Scale a tight-binding model.

    Parameters
    ----------
    tb : dict
        Tight-binding model.
    scale : float
        The scaling factor.

    Returns
    -------
    :
        Scaled tight-binding model.
    """
    return {k: tb.get(k, 0) * scale for k in frozenset(tb)}


def compare_dicts(dict1: dict, dict2: dict, atol: float = 1e-10) -> None:
    """Compare two dictionaries."""
    for key in frozenset(dict1) | frozenset(dict2):
        assert np.allclose(dict1[key], dict2[key], atol=atol)
