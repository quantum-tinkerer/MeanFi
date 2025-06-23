import numpy as np
import pytest
import kwant

import itertools as it

from meanfi.kwant_helper.utils import builder_to_tb, tb_to_builder
from meanfi.tb.utils import generate_tb_keys, guess_tb
from meanfi.tb.tb import compare_dicts

repeat_number = 3


@pytest.mark.parametrize("seed", range(repeat_number))
def test_kwant_conversion(seed):
    """Test conversion between Kwant and meanfi"""
    np.random.seed(seed)
    ndim = np.random.randint(1, 3)
    cutoff = np.random.randint(1, 3)
    sites_in_cell = np.random.randint(1, 4)
    ndof_per_site = [np.random.randint(1, 4) for site in range(sites_in_cell)]
    keyList = generate_tb_keys(cutoff, ndim)

    vecs = np.random.rand(ndim, ndim)

    # set a dummy lattice to read sites from
    lattice = kwant.lattice.general(
        vecs,
        basis=np.random.rand(sites_in_cell, ndim) @ vecs,
        norbs=ndof_per_site,
    )

    dummy_tb = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    for i, sublattice in enumerate(lattice.sublattices):
        dummy_tb[lattice.shape(lambda pos: True, tuple(ndim * [0]))] = np.eye(
            ndof_per_site[i]
        )

    # generate random and generate builder from it
    random_tb = guess_tb(keyList, sum(ndof_per_site))
    random_builder = tb_to_builder(
        random_tb, list(dummy_tb.sites()), dummy_tb.symmetry.periods
    )
    # convert builder back to tb and compare
    random_builder_tb = builder_to_tb(random_builder)
    compare_dicts(random_tb, random_builder_tb)


@pytest.mark.parametrize("seed", range(repeat_number))
def test_builder_order_invariance(seed):
    """Test conversion between Kwant and meanfi for site order"""

    np.random.seed(seed)
    norbs = np.random.randint(2, 4)
    nlen = np.random.randint(2, 4)

    lat = kwant.lattice.square(norbs=norbs)
    builder1 = kwant.Builder(kwant.TranslationalSymmetry((nlen, 0), (0, nlen)))
    builder2 = kwant.Builder(kwant.TranslationalSymmetry((nlen, 0), (0, nlen)))

    sites = [lat(i, j) for i in range(nlen) for j in range(nlen)]
    vals = [np.random.randn(norbs, norbs) for _ in range(len(sites))]

    for idx, site in enumerate(sites):
        builder1[site] = vals[idx]

    indices = np.arange(len(sites))
    np.random.shuffle(indices)
    shuffled_sites = [sites[i] for i in indices]
    shuffled_vals = [vals[i] for i in indices]

    for idx, site in enumerate(shuffled_sites):
        builder2[site] = shuffled_vals[idx]

    tb1 = builder_to_tb(builder1)
    tb2 = builder_to_tb(builder2)

    compare_dicts(tb1, tb2)


@pytest.mark.parametrize("seed", range(repeat_number))
def test_kwant_supercell(seed):
    """Test with Kwant supercell and callable onsite and hoppings."""
    np.random.seed(seed)
    ndim = np.random.randint(1, 3)
    sites_in_cell = np.random.randint(1, 4)
    ndof_per_site = [np.random.randint(1, 4) for site in range(sites_in_cell)]
    n_cells = np.random.randint(1, 4)

    vecs = np.random.rand(ndim, ndim)

    # set a dummy lattice to read sites from
    lattice = kwant.lattice.general(
        vecs,
        basis=np.random.rand(sites_in_cell, ndim) @ vecs,
        norbs=ndof_per_site,
    )

    def random_matrix_kwant_digest(n, m, k):
        matrix = np.zeros((n, m))
        for i in zip(it.product(range(n), range(m))):
            matrix[i[0]] = kwant.digest.uniform(str(n * m * np.prod(i[0]) + k))
        return matrix

    def onsite(site, alpha, beta):
        n = site.family.norbs
        amplitude = alpha * random_matrix_kwant_digest(n, n, 0)
        phase = 1j * 2 * np.pi * beta * random_matrix_kwant_digest(n, n, 1)
        onsite_matrix = amplitude * phase
        onsite_matrix += onsite_matrix.conj().T
        return onsite_matrix

    def hopping(site1, site2, gamma, delta):
        n1 = site1.family.norbs
        n2 = site2.family.norbs
        amplitude = gamma * random_matrix_kwant_digest(n1, n2, 0)
        phase = 1j * 2 * np.pi * delta * random_matrix_kwant_digest(n1, n2, 1)
        hopping_matrix = amplitude * phase
        return hopping_matrix

    random_builder = kwant.Builder(
        kwant.TranslationalSymmetry(*n_cells * lattice.prim_vecs)
    )
    for i, sublattice in enumerate(lattice.sublattices):
        random_builder[lattice.shape(lambda pos: True, tuple(ndim * [0]))] = onsite
    random_builder[lattice.neighbors()] = hopping

    params_num = np.random.rand(4)
    params = dict(
        alpha=params_num[0],
        beta=params_num[1],
        gamma=params_num[2],
        delta=params_num[3],
    )

    random_tb, data = builder_to_tb(random_builder, params=params, return_data=True)
    random_builder_test = tb_to_builder(random_tb, data["sites"], data["periods"])
    for site_pair in zip(it.product(data["sites"], data["sites"])):
        site1, site2 = site_pair[0]
        if site1 == site2:
            assert np.isclose(
                random_builder[site1](
                    site=site1, alpha=params["alpha"], beta=params["beta"]
                ),
                random_builder_test[site1],
            ).all()
        else:
            try:
                assert np.isclose(
                    random_builder[site1, site2](
                        site1, site2, gamma=params["gamma"], delta=params["delta"]
                    ),
                    random_builder_test[site1, site2],
                ).all()
            except KeyError:
                continue
            except:
                raise
