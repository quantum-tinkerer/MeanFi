# %%
import numpy as np
from codes.tb.tb import compareDicts
from codes.kwant_helper import utils    
import itertools as it
from codes.tb.transforms import kfunc2tb, tb2kfunc, tb2kham, tb2khamvector
import pytest

repeatNumber = 10

# %%
ndim = 2
maxOrder = 5
matrixSize = 5
nK = 10


@pytest.mark.repeat(repeatNumber)
def test_fourier():
    keys = [np.arange(-maxOrder + 1, maxOrder) for i in range(ndim)]
    keys = it.product(*keys)
    h_0 = {key: (np.random.rand(matrixSize, matrixSize) - 0.5) * 2 for key in keys}
    kfunc = tb2kfunc(h_0)
    tb_new = kfunc2tb(kfunc, nK, ndim=ndim)
    compareDicts(h_0, tb_new)

@pytest.mark.repeat(repeatNumber):
def test_tbkham_transform(): 
    vectors = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1))
    ndof = 10
    h_0 = utils.generate_guess(vectors, ndof)

    assert np.allclose(tb2kham(h_0, nK=nK, ndim=2), tb2khamvector(h_0, nK=nK, ndim=2))