import numpy as np

from pymf.tb.tb import _tb_type


def expectation_value(density_matrix: _tb_type, observable: _tb_type) -> complex:
    """Compute the expectation value of an observable with respect to a density matrix.

    Parameters
    ----------
    density_matrix :
        Density matrix tight-binding dictionary.
    observable :
        Observable tight-binding dictionary.

    Returns
    -------
    :
        Expectation value.
    """
    return np.sum(
        [
            np.trace(observable[k] @ density_matrix[tuple(-np.array(k))])
            for k in frozenset(density_matrix) & frozenset(observable)
        ]
    )
