import numpy as np
from codes.tb.transforms import tb_to_kfunc
import itertools as it


def compute_gap(h, fermi_energy=0, n=100):
    """
     Compute gap.

     Parameters:
     -----------
     h : dict
     Tight-binding model for which to compute the gap.
    fermi_energy : float
     Fermi energy.
     n : int
     Number of k-points to sample along each dimension.

     Returns:
     --------
     gap : float
     Indirect gap.
    """
    ndim = len(list(h)[0])
    hkfunc = tb_to_kfunc(h)
    k_array = np.linspace(0, 2 * np.pi, n)
    kgrid = list(it.product(*[k_array for i in range(ndim)]))
    kham = np.array([hkfunc(k) for k in kgrid])
    vals = np.linalg.eigvalsh(kham)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)
