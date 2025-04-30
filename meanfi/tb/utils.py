from itertools import product
import numpy as np

from tb.tb import _tb_type
from tb.transforms import ham_fam_to_ort_basis
from params.rparams import projection_to_tb
from qsymm import BlochModel


def generate_tb_vals(
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
        Maximum distance along each dimension to generate tight-binding dictionary keys for.
    dim :
        Dimension of the tight-binding dictionary.

    Returns
    -------
    :
        List of generated tight-binding dictionary keys up to a cutoff.
    """
    return [*product(*([[*range(-cutoff, cutoff + 1)]] * dim))]


def normal_tb(hopdist: int, ndim: int, ndof: int, scale: float = 1) -> _tb_type:
    """Generate tight-binding Hamiltonian dictionary with hoppings up to a maximum `hopdist` in `ndim` dimensions.

    Parameters
    ----------
    hopdist: int
        Maximum hopping distance along each dimension.
    ndim: int
        Dimensions of the tight-binding dictionary.
    ndof: int
        Number of internal degrees of freedom within the unit cell.
    scale: float
        This scales the random values that will be in the Hamiltonian. (`default = 1.0`)

    Returns
    -------
    :
        A random Hermitian tight-binding dictionary.
    """
    # Generate the proper number of keys for all the possible hoppings.
    tb_keys = generate_tb_keys(hopdist, ndim)

    # Generate the dictionary for those keys.
    h_dict = generate_tb_vals(tb_keys, ndof, scale)

    return h_dict


def superc_tb(hopdist: int, ndim: int, ndof: int, scale: float = 1) -> _tb_type:
    """Generate tight-binding superconducting Hamiltonian dictionary with hoppings up to a maximum `hopdist` in `ndim` dimensions.

    Parameters
    ----------
    hopdist: int
        Maximum distance along each dimension.
    ndim: int
        Dimensions of the tight-binding dictionary.
    ndof: int
        Number of internal degrees of freedom within the unit cell per particle. (Electrons and holes)
    scale: float
        This scales the random values that will be in the Hamiltonian. (`default = 1.0`)

    Returns
    -------
    :
        A random hermitian superconducting tight-binding dictionary. (Values with shape `(2 * ndof, 2 * ndof)`)
    """
    # Generate h_0
    h_0 = normal_tb(hopdist, ndim, ndof * 2, scale)
    tau_x = np.kron(np.array([[0, 1], [1, 0]]), np.eye(ndof))

    # Combine these into a superconducting Hamiltonian.
    h_sc_dict = {}
    for key in h_0:
        h_sc_dict[key] = h_0[key] - (tau_x @ h_0[key].conj() @ tau_x)

    return h_sc_dict


def guess_coeffs(tb_basis: dict, scale: float = 1) -> dict:
    """Generate guess coefficient dictionary.

    Parameters
    ----------
    tb_basis: dict
        A tight binding dictionary with a list of basis matrices for every key.
    scale: float
        Scale of the random guess.

    Returns
    -------
    :
        Guess coefficient dictionary with a random guess coefficient for every basis matrix in `tb_basis`.
    """
    guess = {key: scale * np.random.rand(len(tb_basis[key])) for key in tb_basis}

    return guess


def symm_guess_mf(bloch_family: list[BlochModel], scale: float = 1) -> _tb_type:
    """Generates a random meanfield guess using the provided `bloch_family`.

    Parameters
    ----------
    bloch_family : list[BlochModel]
        A list of `qsymm` `BlochModel`s generated with `qsymm.bloch_family(..., bloch_model=True)` or `meanfi.tb.transforms.tb_to_ham_fam`.
    scale: float
        Scale of the random guess.

    Returns
    -------
        A random meanfield guess.
    """
    ham_basis = ham_fam_to_ort_basis(bloch_family)
    random_coeffs = guess_coeffs(ham_basis, scale)
    mf_guess = projection_to_tb(random_coeffs, ham_basis)

    return mf_guess
