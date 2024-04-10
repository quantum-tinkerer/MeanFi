import numpy as np
from codes.tb.transforms import tb2kfunc
import itertools as it

def compute_gap(h, E_F=0, n=100):
    """
    Compute gap.

    Parameters:
    -----------
    h : dict
    Tight-binding model for which to compute the gap.
    E_F : float
    Fermi energy.
    n : int
    Number of k-points to sample along each dimension.

    Returns:
    --------
    gap : float
    Indirect gap.
    """
    ndim = len(list(h)[0])
    hkfunc = tb2kfunc(h)
    kArray = np.linspace(0, 2*np.pi, n)
    kGrid = list(it.product(*[kArray for i in range(ndim)]))
    kham = np.array([hkfunc(k) for k in kGrid]) 
    vals = np.linalg.eigvalsh(kham)

    emax = np.max(vals[vals <= E_F])
    emin = np.min(vals[vals > E_F])
    return np.abs(emin - emax)