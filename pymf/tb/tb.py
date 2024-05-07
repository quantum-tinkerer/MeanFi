import numpy as np

_tb_type = dict[tuple[()] | tuple[int, ...], np.ndarray]


def add_tb(tb1: _tb_type, tb2: _tb_type) -> _tb_type:
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


def scale_tb(tb: _tb_type, scale: float) -> _tb_type:
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
