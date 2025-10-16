from itertools import product
from collections import defaultdict
from typing import Callable, Optional
import inspect

import numpy as np
import kwant
from kwant.builder import Site
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

    sorted_sites = sorted(builder.sites())
    idx_by_site = {site: idx for idx, site in enumerate(sorted_sites)}
    norbs_list = [site.family.norbs for site in sorted_sites]

    if any(norbs is None for norbs in norbs_list):
        raise ValueError("Number of orbitals must be specified for all sites.")

    offsets = np.cumsum([0] + norbs_list)

    tb_norbs = sum(norbs_list)
    tb_shape = (tb_norbs, tb_norbs)
    onsite_idx = tuple([0] * dims)
    h_0 = defaultdict(lambda: np.zeros(tb_shape, dtype=complex))

    onsite = h_0[onsite_idx]
    for site, val in builder.site_value_pairs():
        if callable(val):
            param_keys = inspect.getfullargspec(val).args[1:]
            try:
                val = val(site, *[params[key] for key in param_keys])
            except KeyError as key:
                raise KeyError(f"Parameter {key} not found in params.")

        site_idx = idx_by_site[site]
        onsite[
            offsets[site_idx] : offsets[site_idx + 1],
            offsets[site_idx] : offsets[site_idx + 1],
        ] = val

    for (site1, site2), val in builder.hopping_value_pairs():
        if callable(val):
            param_keys = inspect.getfullargspec(val).args[2:]
            try:
                val = val(site1, site2, *[params[key] for key in param_keys])
            except KeyError as key:
                raise KeyError(f"Parameter {key} not found in params.")

        site2_dom = builder.symmetry.which(site2)
        site2_fd = builder.symmetry.to_fd(site2)

        site1_idx, site2_idx = [idx_by_site[site1], idx_by_site[site2_fd]]
        to_slice, from_slice = [
            slice(offsets[idx], offsets[idx + 1]) for idx in [site1_idx, site2_idx]
        ]

        h_0[site2_dom][to_slice, from_slice] = val
        h_0[-site2_dom][from_slice, to_slice] = val.T.conj()

    if return_data:
        data = {}
        data["periods"] = prim_vecs
        data["sites"] = list(idx_by_site.keys())
        return h_0, data
    else:
        return h_0


def tb_to_builder(
    h_0: _tb_type, sites_list: list[Site, ...], periods: np.ndarray
) -> kwant.builder.Builder:
    """
    Construct a `kwant.builder.Builder` from a tight-binding dictionary.

    Parameters
    ----------
    h_0 :
        Tight-binding dictionary.
    sites_list :
        List of sites in the builder's unit cell.
    periods :
        2d array with periods of the translational symmetry.

    Returns
    -------
    :
        `kwant.builder.Builder` that corresponds to the tight-binding dictionary.
    """
    if periods == ():
        builder = kwant.Builder()
    else:
        builder = kwant.Builder(kwant.TranslationalSymmetry(*periods))
    onsite_idx = tuple([0] * len(list(h_0)[0]))

    sites_list = sorted(sites_list)
    norbs_list = [site.family.norbs for site in sites_list]
    norbs_list = [1 if norbs is None else norbs for norbs in norbs_list]

    def site_to_tbIdxs(site):
        site_idx = sites_list.index(site)
        return (np.sum(norbs_list[:site_idx]) + range(norbs_list[site_idx])).astype(int)

    # assemble the sites first
    for site in sites_list:
        tb_idxs = site_to_tbIdxs(site)
        value = h_0[onsite_idx][
            tb_idxs[0] : tb_idxs[-1] + 1, tb_idxs[0] : tb_idxs[-1] + 1
        ]
        builder[site] = value

    # connect hoppings within the unit-cell
    for site1, site2 in product(sites_list, sites_list):
        if site1 == site2:
            continue
        tb_idxs1 = site_to_tbIdxs(site1)
        tb_idxs2 = site_to_tbIdxs(site2)
        value = h_0[onsite_idx][
            tb_idxs1[0] : tb_idxs1[-1] + 1, tb_idxs2[0] : tb_idxs2[-1] + 1
        ]
        if np.all(value == 0):
            continue
        builder[(site1, site2)] = value

    # connect hoppings between unit-cells
    for key in h_0:
        if key == onsite_idx:
            continue
        for site1, site2_fd in product(sites_list, sites_list):
            site2 = builder.symmetry.act(key, site2_fd)
            tb_idxs1 = site_to_tbIdxs(site1)
            tb_idxs2 = site_to_tbIdxs(site2_fd)
            value = h_0[key][
                tb_idxs1[0] : tb_idxs1[-1] + 1, tb_idxs2[0] : tb_idxs2[-1] + 1
            ]
            if np.all(value == 0):
                continue
            builder[(site1, site2)] = value
    return builder


def build_interacting_syst(
    builder: kwant.builder.Builder,
    lattice: kwant.lattice.Polyatomic,
    func_onsite: Callable,
    func_hop: Optional[Callable] = None,
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
    if func_hop is not None:
        for neighbors in range(max_neighbor + 1):
            int_builder[
                [hop for hop in lattice.neighbors(neighbors) if hop.family_a != hop.family_b]
            ] = func_hop
    return int_builder
