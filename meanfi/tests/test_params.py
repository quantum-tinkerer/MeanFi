# %%
import pytest
import numpy as np
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.tb.tb import compare_dicts
from meanfi.tb.utils import generate_tb_keys
from meanfi import guess_tb

repeat_number = 10
# %%
ndof = 10
vectors = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1))


@pytest.mark.parametrize("seed", range(repeat_number))
def test_parametrisation(seed):
    """Test the parametrisation of the tight-binding model."""
    np.random.seed(seed)
    for ndim in np.arange(4):
        cutoff = np.random.randint(0, 4)
        ndof = np.random.randint(1, 10)

        keys = generate_tb_keys(cutoff, ndim)
        mf_guess = guess_tb(keys, ndof)

        mf_params = tb_to_rparams(mf_guess)
        mf_new = rparams_to_tb(mf_params, list(mf_guess), ndof)
        compare_dicts(mf_guess, mf_new)
