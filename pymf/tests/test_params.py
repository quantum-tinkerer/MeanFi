# %%
import pytest

from pymf.params.rparams import rparams_to_tb, tb_to_rparams
from pymf.tb.tb import compare_dicts
from pymf.tb.utils import generate_guess

repeat_number = 10

# %%
ndof = 10
vectors = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1))


@pytest.mark.repeat(repeat_number)
def test_parametrisation():
    """Test the parametrisation of the tight-binding model."""
    mf_guess = generate_guess(vectors, ndof)
    mf_params = tb_to_rparams(mf_guess)
    mf_new = rparams_to_tb(mf_params, vectors, ndof)
    compare_dicts(mf_guess, mf_new)
