import numpy as np


def compare_dicts(dict1: dict, dict2: dict, atol: float = 1e-10) -> None:
    for key in frozenset(dict1) | frozenset(dict2):
        assert np.allclose(dict1[key], dict2[key], atol=atol)
