# %%
import numpy as np
from codes.tb.tb import compare_dicts
import itertools as it
from codes.tb.transforms import tb_to_khamvector, ifftn_to_tb
from scipy.fftpack import ifftn
import pytest

repeat_number = 10

# %%
ndim = 2
max_order = 5
matrix_size = 5
nk = 10

@pytest.mark.repeat(repeat_number)
def test_fourier():
    keys = [np.arange(-max_order + 1, max_order) for i in range(ndim)]
    keys = it.product(*keys)
    h_0 = {key: (np.random.rand(matrix_size, matrix_size) - 0.5) * 2 for key in keys}
    kham = tb_to_khamvector(h_0, nk=nk)
    tb_new = ifftn_to_tb(ifftn(kham, axes=np.arange(ndim)))
    compare_dicts(h_0, tb_new)