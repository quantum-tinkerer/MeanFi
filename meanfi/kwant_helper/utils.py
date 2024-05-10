from itertools import product
from typing import Callable

import numpy as np
from scipy.sparse import coo_array
import kwant
import kwant.lattice
import kwant.builder

from meanfi.tb.tb import _tb_type


def builder_to_tb(
    builder: kwant.builder.Builder, params: dict = {}, return_data: bool = False
) -> _tb_type:
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
    prim_vecs = builder.symmetry.periods
    dims = len(prim_vecs)
    sites_list = [*builder.sites()]
    norbs_list = [site.family.norbs for site in builder.sites()]
    norbs_list = [1 if norbs is None else norbs for norbs in norbs_list]

    tb_norbs = sum(norbs_list)
    tb_shape = (tb_norbs, tb_norbs)
    onsite_idx = tuple([0] * dims)
    h_0 = {}

    def _parse_val(val):
        if callable(val):
            param_keys = val.__code__.co_varnames[1:]
            try:
                val = val(site, *[params[key] for key in param_keys])
            except KeyError as key:
                raise KeyError(f"Parameter {key} not found in params.")
        return val

    for site, val in builder.site_value_pairs():
        site_idx = sites_list.index(site)
        tb_idx = np.sum(norbs_list[:site_idx]) + range(norbs_list[site_idx])
        row, col = np.array([*product(tb_idx, tb_idx)]).T

        data = np.array(_parse_val(val)).flatten()
        onsite_value = coo_array((data, (row, col)), shape=tb_shape).toarray()

        if onsite_idx in h_0:
            h_0[onsite_idx] += onsite_value
        else:
            h_0[onsite_idx] = onsite_value

    for (site1, site2), val in builder.hopping_value_pairs():
        site2_dom = builder.symmetry.which(site2)
        site2_fd = builder.symmetry.to_fd(site2)

        site1_idx, site2_idx = np.array(
            [sites_list.index(site1), sites_list.index(site2_fd)]
        )
        tb_idx1, tb_idx2 = [
            np.sum(norbs_list[:site1_idx]) + range(norbs_list[site1_idx]),
            np.sum(norbs_list[:site2_idx]) + range(norbs_list[site2_idx]),
        ]
        row, col = np.array([*product(tb_idx1, tb_idx2)]).T
        data = np.array(_parse_val(val)).flatten()
        hopping_value = coo_array((data, (row, col)), shape=tb_shape).toarray()

        hop_key = tuple(site2_dom)
        hop_key_back = tuple(-site2_dom)
        if hop_key in h_0:
            h_0[hop_key] += hopping_value
            if np.linalg.norm(site2_dom) == 0:
                h_0[hop_key] += hopping_value.conj().T
            else:
                h_0[hop_key_back] += hopping_value.conj().T
        else:
            h_0[hop_key] = hopping_value
            if np.linalg.norm(site2_dom) == 0:
                h_0[hop_key] += hopping_value.conj().T
            else:
                h_0[hop_key_back] = hopping_value.conj().T

    if return_data:
        data = {}
        data["norbs"] = norbs_list
        data["positions"] = [site.pos for site in sites_list]
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
