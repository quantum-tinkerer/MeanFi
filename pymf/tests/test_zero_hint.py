# %%
import numpy as np
import pytest

from pymf.tb import utils
from pymf.tb.tb import compare_dicts
from pymf import Model, solver, generate_guess, add_tb, calculate_fermi_energy

# %%
repeat_number = 10


# %%
@pytest.mark.parametrize("seed", range(repeat_number))
def test_zero_hint(seed):
    """Test the zero interaction case for the tight-binding model."""
    np.random.seed(seed)

    cutoff = np.random.randint(1, 4)
    dim = np.random.randint(0, 3)
    ndof = np.random.randint(2, 10)
    filling = np.random.randint(1, ndof)
    random_hopping_vecs = utils.generate_tb_keys(cutoff, dim)

    zero_key = tuple([0] * dim)
    h_0_random = generate_guess(random_hopping_vecs, ndof, scale=1)
    h_int_only_phases = generate_guess(random_hopping_vecs, ndof, scale=0)
    guess = generate_guess(random_hopping_vecs, ndof, scale=1)
    model = Model(h_0_random, h_int_only_phases, filling=filling)

    mf_sol = solver(model, guess, nk=40, optimizer_kwargs={"M": 0, "f_tol": 1e-10})
    h_fermi = calculate_fermi_energy(mf_sol, filling=filling, nk=20)
    mf_sol[zero_key] -= h_fermi * np.eye(mf_sol[zero_key].shape[0])

    compare_dicts(add_tb(mf_sol, h_0_random), h_0_random, atol=1e-10)