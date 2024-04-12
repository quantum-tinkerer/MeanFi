# %%
from codes.params.rparams import mf_to_rparams, rparams_to_mf
from codes.kwant_helper.utils import generate_guess
from codes.tb.tb import compare_dicts
import pytest

repeat_number = 10

# %%
ndof = 10
vectors = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1))


@pytest.mark.repeat(repeat_number)
def test_parametrisation():
    mf_guess = generate_guess(vectors, ndof)
    mf_params = mf_to_rparams(mf_guess)
    mf_new = rparams_to_mf(mf_params, vectors, ndof)
    compare_dicts(mf_guess, mf_new)
