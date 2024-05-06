import numpy as np

tb_type = dict[tuple[None] | tuple[int, ...], np.ndarray]


def add_tb(tb1: tb_type, tb2: tb_type) -> tb_type:
    """Add up two tight-binding dictionaries together.

    Parameters
    ----------
    tb1 :
        Tight-binding dictionary.
    tb2 :
        Tight-binding dictionary.

    Returns
    -------
    :
        Sum of the two tight-binding dictionaries.
    """
    return {k: tb1.get(k, 0) + tb2.get(k, 0) for k in frozenset(tb1) | frozenset(tb2)}


def scale_tb(tb: tb_type, scale: float) -> tb_type:
    """Scale a tight-binding dictionary by a constant.

    Parameters
    ----------
    tb :
        Tight-binding dictionary.
    scale :
        Constant to scale the tight-binding dictionary by.

    Returns
    -------
    :
        Scaled tight-binding dictionary.
    """
    return {k: tb.get(k, 0) * scale for k in frozenset(tb)}


def compare_dicts(dict1: dict, dict2: dict, atol: float = 1e-10) -> None:
    """Compare two dictionaries."""
    for key in frozenset(dict1) | frozenset(dict2):
        assert np.allclose(dict1[key], dict2[key], atol=atol)
