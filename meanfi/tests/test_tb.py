# %%
import itertools as it

import numpy as np
import pytest
from scipy.fftpack import ifftn

from meanfi.tb.tb import compare_dicts
from meanfi.tb.transforms import ifftn_to_tb, tb_to_kgrid

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
