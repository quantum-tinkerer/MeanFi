# %%
import itertools as it

import numpy as np
import pytest
from scipy.fftpack import ifftn

from meanfi.tb.tb import compare_dicts
from meanfi.tb.utils import guess_tb, generate_tb_keys
from meanfi.tb.transforms import ifftn_to_tb, tb_to_kgrid, tb_to_kfunc

repeat_number = 10

# %%
ndim = 2
max_order = 5
matrix_size = 5
nk = 10


@pytest.mark.parametrize("seed", range(repeat_number))
def test_fourier(seed):
    """Test the Fourier transformation of the tight-binding model."""
    np.random.seed(seed)
    keys = [np.arange(-max_order + 1, max_order) for i in range(ndim)]
    keys = it.product(*keys)
    h_0 = {key: (np.random.rand(matrix_size, matrix_size) - 0.5) * 2 for key in keys}
    kham = tb_to_kgrid(h_0, nk=nk)
    tb_new = ifftn_to_tb(ifftn(kham, axes=np.arange(ndim)))
    compare_dicts(h_0, tb_new)


@pytest.mark.parametrize("seed", range(repeat_number))
def test_kfunc(seed):
    np.random.seed(seed)
    cutoff = np.random.randint(1, 4)
    dim = np.random.randint(1, 3)
    ndof = np.random.randint(2, 10)
    nk = np.random.randint(3, 10)
    random_hopping_vecs = generate_tb_keys(cutoff, dim)
    random_tb = guess_tb(random_hopping_vecs, ndof, scale=1)

    kfunc = tb_to_kfunc(random_tb)
    kham = tb_to_kgrid(random_tb, nk=nk)

    # evaluate kfunc on same grid as kham
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    ks = np.concatenate((ks[nk // 2 :], ks[: nk // 2]), axis=0)
    k_pts = np.tile(ks, dim).reshape(dim, nk)

    ham_kfunc = []
    for k in it.product(*k_pts):
        ham_kfunc.append(kfunc(k))
    ham_kfunc = np.array(ham_kfunc).reshape(*kham.shape)

    assert np.allclose(kham, ham_kfunc)
