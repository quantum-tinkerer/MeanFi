import numpy as np
import kwant
from scipy.sparse import coo_array
from itertools import product
import inspect
from copy import copy


def builder2tb(builder, params={}, return_data=False):
    """
    Constructs a tight-binding model dictionary from a `kwant.Builder`.

    Parameters:
    -----------
    builder : `kwant.Builder`
        Either builder for non-interacting system or interacting Hamiltonian.
    params : dict
        Dictionary of parameters to evaluate the Hamiltonian.
    return_data : bool
        Returns dictionary with sites and number of orbitals per site.

    Returns:
    --------
    h_0 : dict
        Tight-binding model of non-interacting systems.
    data : dict
        Data with sites and number of orbitals. Only if `return_data=True`.
    """
    builder = copy(builder)
    # Extract information from builder
    dims = len(builder.symmetry.periods)
    onsite_idx = tuple([0] * dims)
    h_0 = {}
    sites_list = [*builder.sites()]
    norbs_list = [site[0].norbs for site in builder.sites()]
    positions_list = [site[0].pos for site in builder.sites()]
    norbs_tot = sum(norbs_list)
    # Extract onsite and hopping matrices.
    # Based on `kwant.wraparound.wraparound`
    # Onsite matrices
    for site, val in builder.site_value_pairs():
        site = builder.symmetry.to_fd(site)
        atom = sites_list.index(site)
        row = np.sum(norbs_list[:atom]) + range(norbs_list[atom])
        col = copy(row)
        row, col = np.array([*product(row, col)]).T
        try:
            for arg in inspect.getfullargspec(val).args:
                _params = {}
                if arg in params:
                    _params[arg] = params[arg]
            val = val(site, **_params)
            data = val.flatten()
        except Exception:
            data = val.flatten()
        if onsite_idx in h_0:
            h_0[onsite_idx] += coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
        else:
            h_0[onsite_idx] = coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
    # Hopping matrices
    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        b_dom = builder.symmetry.which(b)
        b_fd = builder.symmetry.to_fd(b)
        atoms = np.array([sites_list.index(a), sites_list.index(b_fd)])
        row, col = [
            np.sum(norbs_list[: atoms[0]]) + range(norbs_list[atoms[0]]),
            np.sum(norbs_list[: atoms[1]]) + range(norbs_list[atoms[1]]),
        ]
        row, col = np.array([*product(row, col)]).T
        try:
            for arg in inspect.getfullargspec(val).args:
                _params = {}
                if arg in params:
                    _params[arg] = params[arg]
            val = val(a, b, **_params)
            data = val.flatten()
        except Exception:
            data = val.flatten()
        if tuple(b_dom) in h_0:
            h_0[tuple(b_dom)] += coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
            if np.linalg.norm(b_dom) == 0:
                h_0[tuple(b_dom)] += (
                    coo_array((data, (row, col)), shape=(norbs_tot, norbs_tot))
                    .toarray()
                    .T.conj()
                )
            else:
                # Hopping vector in the opposite direction
                h_0[tuple(-b_dom)] += (
                    coo_array((data, (row, col)), shape=(norbs_tot, norbs_tot))
                    .toarray()
                    .T.conj()
                )
        else:
            h_0[tuple(b_dom)] = coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
            if np.linalg.norm(b_dom) == 0:
                h_0[tuple(b_dom)] += (
                    coo_array((data, (row, col)), shape=(norbs_tot, norbs_tot))
                    .toarray()
                    .T.conj()
                )
            else:
                h_0[tuple(-b_dom)] = (
                    coo_array((data, (row, col)), shape=(norbs_tot, norbs_tot))
                    .toarray()
                    .T.conj()
                )

    if return_data:
        data = {}
        data["norbs"] = norbs_list
        data["positions"] = positions_list
        return h_0, data
    else:
        return h_0


def build_interacting_syst(builder, lattice, func_onsite, func_hop, max_neighbor=1):
    """
    Construct an auxiliary `kwant` system to build Hamiltonian matrix.

    Parameters:
    -----------
    builder : `kwant.Builder`
        Non-interacting `kwant` system.
    lattice : `kwant.lattice`
        System lattice.
    func_onsite : function
        Onsite function.
    func_hop : function
        Hopping function.
    max_neighbor : int
        Maximal nearest-neighbor order.

    Returns:
    --------
    int_builder : `kwant.Builder`
        Dummy `kwant.Builder` to compute interaction matrix.
    """
    int_builder = kwant.Builder(kwant.TranslationalSymmetry(*builder.symmetry.periods))
    int_builder[builder.sites()] = func_onsite
    for neighbors in range(max_neighbor):
        int_builder[lattice.neighbors(neighbors + 1)] = func_hop
    return int_builder


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


def calc_gap(vals, fermi_energy):
    """
     Compute gap.

     Parameters:
     -----------
     vals : nd-array
         Eigenvalues on a k-point grid.
    fermi_energy : float
         Fermi energy.

     Returns:
     --------
     gap : float
         Indirect gap.
    """
    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)
