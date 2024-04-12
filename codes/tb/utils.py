import numpy as np
from codes.tb.transforms import tb_to_kfunc
import itertools as it
from itertools import product


def generate_guess(vectors, ndof, scale=1):
    """
    vectors : list
        List of hopping vectors.
    ndof : int
        Number internal degrees of freedom (orbitals),
    scale : float
        The scale of the guess. Maximum absolute value of each element of the guess.

    Returns:
    --------
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
    """
    Generates hopping vectors up to a cutoff.

    Parameters:
    -----------
    cutoff : int
        Maximum distance along each direction.
    dim : int
        Dimension of the vectors.

    Returns:
    --------
    List of hopping vectors.
    """
    return [*product(*([[*range(-cutoff, cutoff + 1)]] * dim))]


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
