import inspect
from copy import copy
from itertools import product
from typing import Callable
from pymf.tb.tb import tb_type

import kwant
import kwant.lattice
import kwant.builder
import numpy as np
from scipy.sparse import coo_array


def builder_to_tb(
    builder: kwant.builder.Builder, params: dict = {}, return_data: bool = False
) -> tb_type:
    """Construct a tight-binding dictionary from a `kwant.builder.Builder` system.

    Parameters
    ----------
    builder :
       system to convert to tight-binding dictionary.
    params :
        Dictionary of parameters to evaluate the builder on.
    return_data :
        Returns dictionary with sites and number of orbitals per site.

    Returns
    -------
    :
        Tight-binding dictionary that corresponds to the builder.
    :
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


def build_interacting_syst(
    builder: kwant.builder.Builder,
    lattice: kwant.lattice.Polyatomic,
    func_onsite: Callable,
    func_hop: Callable,
    max_neighbor: int = 1,
) -> kwant.builder.Builder:
    """
    Construct an auxiliary `kwant` system that encodes the interactions.

    Parameters
    ----------
    builder :
        Non-interacting `kwant.builder.Builder` system.
    lattice :
        Lattice of the system.
    func_onsite :
        Onsite interactions function.
    func_hop :
        Hopping/inter unit cell interactions function.
    max_neighbor :
        The maximal number of neighbouring unit cells (along a lattice vector)
        connected by interaction. Interaction goes to zero after this distance.

    Returns
    -------
    :
        Auxiliary `kwant.builder.Builder` that encodes the interactions of the system.
    """
    int_builder = kwant.builder.Builder(
        kwant.lattice.TranslationalSymmetry(*builder.symmetry.periods)
    )
    int_builder[builder.sites()] = func_onsite
    for neighbors in range(max_neighbor):
        int_builder[lattice.neighbors(neighbors + 1)] = func_hop
    return int_builder
