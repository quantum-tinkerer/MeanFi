import numpy as np


def add_tb(tb1, tb2):
    """
    Add up two tight-binding models together.

    Parameters:
    -----------
    tb1 : dict
        Tight-binding model.
    tb2 : dict
        Tight-binding model.

    Returns:
    --------
    dict
        Sum of the two tight-binding models.
    """
    return {k: tb1.get(k, 0) + tb2.get(k, 0) for k in frozenset(tb1) | frozenset(tb2)}


def scale_tb(tb, scale):
    """
    Scale a tight-binding model.

    Parameters:
    -----------
    tb : dict
        Tight-binding model.
    scale : float
        The scaling factor.

    Returns:
    --------
    dict
        Scaled tight-binding model.
    """
    return {k: tb.get(k, 0) * scale for k in frozenset(tb)}


def compare_dicts(dict1, dict2, atol=1e-10):
    for key in dict1.keys():
        assert np.allclose(dict1[key], dict2[key], atol=atol)
