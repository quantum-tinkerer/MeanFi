from itertools import product
import numpy as np

from meanfi.tb.ops import _tb_type
from meanfi.tb.transforms import tb_to_kgrid


def generate_tb_keys(cutoff: int, dim: int) -> list[tuple[None] | tuple[int, ...]]:
    """Generate tight-binding dictionary keys up to a cutoff.

    Parameters
    ----------
    cutoff :
        Maximum distance along each dimension to generate tight-bindign dictionary keys for.
    dim :
        Dimension of the tight-binding dictionary.

    Returns
    -------
    :
        List of generated tight-binding dictionary keys up to a cutoff.
    """
    return [*product(*([[*range(-cutoff, cutoff + 1)]] * dim))]


def fermi_energy(tb: _tb_type, filling: float, nk: int = 100):
    """
    Calculate the Fermi energy of a given tight-binding dictionary.

    Parameters
    ----------
    tb :
        Tight-binding dictionary.
    filling :
        Number of particles in a unit cell.
        Used to determine the Fermi level.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        Fermi energy.
    """
    kham = tb_to_kgrid(tb, nk)
    vals = np.linalg.eigvalsh(kham)
    flat = np.sort(vals.reshape(-1))
    n_kpoints = vals.shape[0] if vals.ndim == 2 else int(np.prod(vals.shape[:-1]))
    idx = int(np.clip(np.ceil(filling * n_kpoints) - 1, 0, flat.size - 1))
    return float(flat[idx])
