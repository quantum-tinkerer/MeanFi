# %%
import numpy as np
from codes.tb.tb import compare_dicts
import itertools as it
from codes.tb.utils import generate_guess
from codes.tb.transforms import kfunc_to_tb, tb_to_kfunc, tb_to_kham, tb_to_khamvector
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
    kfunc = tb_to_kfunc(h_0)
    tb_new = kfunc_to_tb(kfunc, nk, ndim=ndim)
    compare_dicts(h_0, tb_new)


@pytest.mark.repeat(repeat_number)
def test_tbkham_transform():
    vectors = (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
        (-1, -1),
    )
    ndof = 10
    h_0 = generate_guess(vectors, ndof)

    assert np.allclose(
        tb_to_kham(h_0, nk=nk, ndim=2), tb_to_khamvector(h_0, nk=nk, ndim=2)
    )
