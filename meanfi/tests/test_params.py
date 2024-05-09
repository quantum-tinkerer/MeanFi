# %%
import pytest
import numpy as np
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.tb.tb import compare_dicts
from meanfi import guess_tb

repeat_number = 10

# %%
ndof = 10
vectors = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1))


@pytest.mark.parametrize("seed", range(repeat_number))
def test_parametrisation(seed):
    """Test the parametrisation of the tight-binding model."""
    np.random.seed(seed)
    mf_guess = guess_tb(vectors, ndof)
    mf_params = tb_to_rparams(mf_guess)
    mf_new = rparams_to_tb(mf_params, vectors, ndof)
    compare_dicts(mf_guess, mf_new)
