import numpy as np

def expectationValue(densityMatrix, observable):
    """
    Compute the expectation value of an observable with respect to a density matrix.

    Parameters
    ----------
    densityMatrix : dict
        Density matrix in tight-binding format.
    observable : dict
        Observable in tight-binding format.

    Returns
    -------
    float
        Expectation value.
    """
    return np.sum([np.trace(densityMatrix[k] @ observable[k]) for k in frozenset(densityMatrix) & frozenset(observable)])