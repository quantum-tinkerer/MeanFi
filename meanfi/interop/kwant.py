"""Conversions between Kwant builders and dense or sparse tight-binding blocks."""

from collections import defaultdict
from typing import Callable
import inspect

import numpy as np
import kwant
from scipy import sparse as scipy_sparse
from kwant.builder import Site

from meanfi.tb.ops import _tb_type


def _site_slices(sites):
    norbs = [site.family.norbs for site in sites]
    if any(n is None for n in norbs):
        raise ValueError("Number of orbitals must be specified for all sites.")
    offsets = np.cumsum([0, *norbs])
    return {
        site: slice(start, stop)
        for site, start, stop in zip(sites, offsets[:-1], offsets[1:], strict=True)
    }


def builder_to_tb(
    builder: kwant.Builder, *, params=None, return_data=False, sparse=False
):
    """Evaluate a builder into tight-binding blocks.

    ``sparse=True`` assembles CSR output without dense unit-cell matrices.
    ``return_data=True`` also returns sites and symmetry periods for inversion.
    Scalars on multi-orbital sites are interpreted as multiples of the identity.
    """
    sites = sorted(builder.sites())
    if not sites:
        raise ValueError("builder must contain at least one site")
    slices = _site_slices(sites)
    size = slices[sites[-1]].stop
    periods = getattr(builder.symmetry, "periods", ())
    onsite = (0,) * len(periods)
    params = {} if params is None else params
    allocate = scipy_sparse.lil_matrix if sparse else np.zeros
    blocks = defaultdict(lambda: allocate((size, size), dtype=complex))
    blocks[onsite]  # Include the local block even when all onsite values are zero.

    def value_at(value, *sites):
        if callable(value):
            names = inspect.getfullargspec(value).args[len(sites) :]
            value = value(*sites, *[params[name] for name in names])
        return value

    def put(key, row, col, value):
        if np.ndim(value) == 0 and not scipy_sparse.issparse(value):
            value = value * np.eye(row.stop - row.start, col.stop - col.start)
        blocks[key][row, col] = value

    for site, value in builder.site_value_pairs():
        put(onsite, slices[site], slices[site], value_at(value, site))
    for (site1, site2), value in builder.hopping_value_pairs():
        key = tuple(builder.symmetry.which(site2))
        row, col = slices[site1], slices[builder.symmetry.to_fd(site2)]
        value = value_at(value, site1, site2)
        if not scipy_sparse.issparse(value):
            value = np.asarray(value)
        put(key, row, col, value)
        put(tuple(-r for r in key), col, row, value.conj().T)

    tb = {key: matrix.tocsr() if sparse else matrix for key, matrix in blocks.items()}
    return (tb, {"periods": periods, "sites": sites}) if return_data else tb


def tb_to_builder(
    h_0: _tb_type, sites_list: list[Site], periods: np.ndarray
) -> kwant.Builder:
    """Reconstruct a builder from blocks and ``builder_to_tb`` metadata.

    Only site pairs with nonzero entries are visited. Sparse inputs are
    densified one site-to-site block at a time, as required by Kwant.
    """
    builder = (
        kwant.Builder(kwant.TranslationalSymmetry(*periods))
        if len(periods)
        else kwant.Builder()
    )
    sites = sorted(sites_list)
    slices = _site_slices(sites)
    onsite = (0,) * len(periods)
    blocks = {key: scipy_sparse.csr_matrix(matrix) for key, matrix in h_0.items()}
    for site in sites:
        section = slices[site]
        builder[site] = blocks[onsite][section, section].toarray()
    site_by_orbital = np.repeat(
        np.arange(len(sites)), [site.family.norbs for site in sites]
    )
    for key, matrix in blocks.items():
        rows, cols = matrix.nonzero()
        pairs = set(zip(site_by_orbital[rows], site_by_orbital[cols], strict=True))
        for row, col in sorted(pairs):
            if key == onsite and row == col:
                continue
            source, target = sites[row], sites[col]
            value = matrix[slices[source], slices[target]].toarray()
            builder[source, builder.symmetry.act(key, target)] = value
    return builder


def build_interacting_syst(
    builder: kwant.builder.Builder,
    lattice: kwant.lattice.Polyatomic,
    func_onsite: Callable,
    func_hop: Callable | None = None,
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
    int_builder = kwant.Builder(builder.symmetry)
    int_builder[builder.sites()] = func_onsite
    if func_hop is not None:
        for neighbors in range(max_neighbor + 1):
            hops = lattice.neighbors(neighbors)
            if neighbors == 0:
                hops = [hop for hop in hops if hop.family_a != hop.family_b]
            int_builder[hops] = func_hop
    return int_builder
