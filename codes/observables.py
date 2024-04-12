import numpy as np


def expectation_value(density_matrix, observable):
    """
    Compute the expectation value of an observable with respect to a density matrix.

    Parameters
    ----------
    density_matrix : dict
        Density matrix in tight-binding format.
    observable : dict
        Observable in tight-binding format.

    Returns
    -------
    float
        Expectation value.
    """
    return np.sum(
        [
            np.trace(density_matrix[k] @ observable[k])
            for k in frozenset(density_matrix) & frozenset(observable)
        ]
    )
