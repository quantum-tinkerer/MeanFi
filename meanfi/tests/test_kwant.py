import numpy as np
import pytest
import kwant

from meanfi.kwant_helper.utils import builder_to_tb, tb_to_builder
from meanfi.tb.utils import generate_tb_keys, guess_tb
from meanfi.tb.tb import compare_dicts

repeat_number = 3


@pytest.mark.parametrize("seed", range(repeat_number))
def test_kwant_conversion(seed):
    """Test the gap prediction for the Hubbard model."""
    np.random.seed(seed)
    ndim = np.random.randint(1, 3)
    cutoff = np.random.randint(1, 3)
    sites_in_cell = np.random.randint(1, 4)
    ndof_per_site = np.random.randint(1, 5)
    keyList = generate_tb_keys(cutoff, ndim)

    # set a dummy lattice to read sites from
    lattice = kwant.lattice.general(np.eye(ndim), norbs=ndof_per_site)
    dummy_tb = kwant.Builder(
        kwant.TranslationalSymmetry(*sites_in_cell * lattice.prim_vecs)
    )
    for site in range(sites_in_cell):
        dummy_tb[lattice(site, *[0 for _ in range(ndim - 1)])] = (
            np.eye(ndof_per_site) * 2
        )

    # generate random and generate builder from it
    random_tb = guess_tb(keyList, ndof_per_site * sites_in_cell)
    random_builder = tb_to_builder(
        random_tb, list(dummy_tb.sites()), dummy_tb.symmetry.periods
    )
    # convert builder back to tb and compare
    random_builder_tb = builder_to_tb(random_builder)
    compare_dicts(random_tb, random_builder_tb)
