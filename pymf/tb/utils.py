from itertools import product
import numpy as np

from pymf.tb.tb import tb_type
from pymf.mf import fermi_on_grid
from pymf.tb.transforms import tb_to_khamvector


def generate_guess(
    vectors: list[tuple[None] | tuple[int, ...]], ndof: int, scale: float = 1
) -> tb_type:
    """Generate guess for a tight-binding model.

    Parameters
    ----------
    vectors :
        List of hopping vectors.
    ndof :
        Number internal degrees of freedom (e.g. orbitals, spin, sublattice),
    scale :
        Scale of the random guess. Default is 1.
    Returns
    -------
    :
        Guess in the form of a tight-binding model.
    """
    guess = {}
    for vector in vectors:
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


def generate_vectors(cutoff: int, dim: int) -> list[tuple[None] | tuple[int, ...]]:
    """Generate hopping vectors up to a cutoff.

    Parameters
    ----------
    cutoff :
        Maximum distance along each direction.
    dim :
        Dimension of the vectors.

    Returns
    -------
    List of hopping vectors.
    """
    return [*product(*([[*range(-cutoff, cutoff + 1)]] * dim))]


def calculate_fermi_energy(tb: tb_type, filling: float, nk: int = 100):
    """Calculate the Fermi energy for a given filling."""
    kham = tb_to_khamvector(tb, nk, ks=None)
    vals = np.linalg.eigvalsh(kham)
    return fermi_on_grid(vals, filling)
