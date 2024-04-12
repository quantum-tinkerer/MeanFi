import numpy as np
import kwant
from scipy.sparse import coo_array
from itertools import product
import inspect
from copy import copy


def builder_to_tb(builder, params={}, return_data=False):
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
