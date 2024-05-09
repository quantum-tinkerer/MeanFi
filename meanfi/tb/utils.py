from itertools import product
import numpy as np

from meanfi.tb.tb import _tb_type
from meanfi.mf import fermi_on_kgrid
from meanfi.tb.transforms import tb_to_kgrid


def guess_tb(
    tb_keys: list[tuple[None] | tuple[int, ...]], ndof: int, scale: float = 1
) -> _tb_type:
    """Generate hermitian guess tight-binding dictionary.

    Parameters
    ----------
    tb_keys :
       List of hopping vectors (tight-binding dictionary keys) the guess contains.
    ndof :
        Number internal degrees of freedom within the unit cell.
    scale :
        Scale of the random guess.
    Returns
    -------
    :
        Hermitian guess tight-binding dictionary.
    """
    guess = {}
    for vector in tb_keys:
        if vector not in guess.keys():
            amplitude = scale * np.random.rand(ndof, ndof)
            phase = 2 * np.pi * np.random.rand(ndof, ndof)
            rand_hermitian = amplitude * np.exp(1j * phase)
            if np.linalg.norm(np.array(vector)) == 0:
                rand_hermitian += rand_hermitian.T.conj()
                rand_hermitian /= 2
                guess[vector] = rand_hermitian
            else:
                guess[vector] = rand_hermitian
                guess[tuple(-np.array(vector))] = rand_hermitian.T.conj()

    return guess


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
    return fermi_on_kgrid(vals, filling)
