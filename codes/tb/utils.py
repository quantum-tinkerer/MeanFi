import numpy as np
from scipy.integrate import quad_vec
from functools import partial

def quad_vecNDim(f, a, b, ndim, **kwargs):
    """
    Integrate f over the n-dimensional hypercube [a, b]^n.

    Parameters
    ----------
    f : callable
        Vector function to be integrated with input tuple of length n.
    a : float
        Lower bound of integration.
    b : float
        Upper bound of integration.
    n : int
        Dimension of the hypercube.
    kwargs : dict, optional
        Extra keyword arguments to pass to quad.
    """
    if ndim == 1:
        return quad_vec(f, a, b, **kwargs)
    if ndim == 2:
        _f = lambda x, y : f((x, y))
        _fx = lambda x : quad_vec(partial(_f, x), a, b, **kwargs)[0]
        return quad_vec(_fx, a, b, **kwargs)[0]
    elif ndim > 2:
        raise NotImplementedError("n > 2 not implemented")
 

# test part TODO separate it out to a test file
def gaussianTestFunc(vec):
    x = np.sum(np.array(vec)**2)
    return np.diag([np.exp(-x), np.exp(-2*x)])

assert np.allclose(quad_vecNDim(gaussianTestFunc, -np.inf, np.inf, 2), np.diag([np.pi, np.pi/2]))