import numpy as np
from pymf.tb.tb import tb_type


def expectation_value(density_matrix: tb_type, observable: tb_type) -> complex:
    """Compute the expectation value of an observable with respect to a density matrix.

    Parameters
    ----------
    density_matrix :
        Density matrix in tight-binding format.
    observable :
        Observable in tight-binding format.

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
