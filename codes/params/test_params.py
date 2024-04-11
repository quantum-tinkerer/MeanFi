# %%
from codes.params.rparams import mf2rParams, rParams2mf
from codes.kwant_helper.utils import generate_guess
from codes.tb.tb import compareDicts
import pytest
repeatNumber = 10

# %%
ndof = 10
vectors = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1))

@pytest.mark.repeat(repeatNumber)
def test_parametrisation():
    mf_guess = generate_guess(vectors, ndof)
    mf_params = mf2rParams(mf_guess)
    mf_new = rParams2mf(mf_params, vectors, ndof)
    compareDicts(mf_guess, mf_new)