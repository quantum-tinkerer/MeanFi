# %%
import numpy as np
import pytest

from meanfi.mf import add_tb, density_matrix
from meanfi.tb.utils import generate_tb_keys, guess_tb


repeat_number = 5


# %%
def minimizer_offset(
    cutoff: int,
    ndim: int,
    ndof: int,
    filling: float,
    nk: int,
    kT: float,
    f_random: float,
):
    keys = generate_tb_keys(cutoff, ndim)
    h_0 = guess_tb(keys, ndof)

    f_level = density_matrix(h_0, filling, nk, kT)[1]

    # Shift the Hamiltonian.
    _shift = {(0,) * ndim: -f_level * np.eye(ndof)}
    h_shift = add_tb(h_0, _shift)

    # Generate an offset Hamiltonian.

    _offset = {(0,) * ndim: f_random * np.eye(ndof)}
    h_offset = add_tb(h_shift, _offset)

    # Compute f_offset for the offset Hamiltonian.
    f_offset = density_matrix(h_offset, filling, nk, kT)[1]

    assert np.allclose(f_random, f_offset, kT / 2, kT / 2)


# %%
@pytest.mark.parametrize("seed", range(repeat_number))
def test_minimizer_consistency(seed):
    np.random.seed(seed)
    ndim = np.random.randint(1, 4)
    ndof = np.random.randint(1, 8)
    cutoff = np.random.randint(1, 5)
    nk = np.random.randint(10, 100)
    filling = np.random.uniform(0, ndof)
    kT = np.random.uniform(0, 1e-2)
    f_random = np.random.uniform(-3, 3)
    minimizer_offset(cutoff, ndim, ndof, filling, nk, kT, f_random)
