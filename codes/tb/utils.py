from itertools import product

import numpy as np

from codes.mf import fermi_on_grid
from codes.tb.transforms import tb_to_khamvector


def generate_guess(vectors, ndof, scale=1):
    """Generate guess for a tight-binding model.

    Parameters
    ----------
    vectors : list
        List of hopping vectors.
    ndof : int
        Number internal degrees of freedom (orbitals),
    scale : float
        The scale of the guess. Maximum absolute value of each element of the guess.

    Returns
    -------
    guess : tb dictionary
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


def generate_vectors(cutoff, dim):
    """Generate hopping vectors up to a cutoff.

    Parameters
    ----------
    cutoff : int
        Maximum distance along each direction.
    dim : int
        Dimension of the vectors.

    Returns
    -------
    List of hopping vectors.
    """
    return [*product(*([[*range(-cutoff, cutoff + 1)]] * dim))]


def compute_gap(tb, fermi_energy=0, nk=100):
    """Compute gap.

    Parameters
    ----------
    tb : dict
        Tight-binding model for which to compute the gap.
    fermi_energy : float
     Fermi energy.
    nk : int
     Number of k-points to sample along each dimension.

    Returns
    -------
     gap : float
     Indirect gap.
    """
    kham = tb_to_khamvector(tb, nk, ks=None)
    vals = np.linalg.eigvalsh(kham)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)


def calculate_fermi_energy(tb, filling, nk=100):
    """Calculate the Fermi energy for a given filling."""
    kham = tb_to_khamvector(tb, nk, ks=None)
    vals = np.linalg.eigvalsh(kham)
    return fermi_on_grid(vals, filling)
