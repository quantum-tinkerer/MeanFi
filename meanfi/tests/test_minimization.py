# %%
import numpy as np
import pytest

from meanfi.mf import add_tb, fermi_level
from meanfi.tb.utils import generate_tb_keys, generate_tb_vals

repeat_number = 5


# %%
def minimizer_offset(
    cutoff: int,
    ndim: int,
    ndof: int,
    target_charge: float,
    charge_op: np.ndarray,
    nk: int,
    kT: float,
    f_random: float,
):
    keys = generate_tb_keys(cutoff, ndim)
    h_0 = generate_tb_vals(keys, ndof)

    f_level = fermi_level(h_0, charge_op, target_charge, kT, nk, ndim)

    # Shift the Hamiltonian.
    _shift = {(0,) * ndim: -f_level * charge_op}
    h_shift = add_tb(h_0, _shift)

    # Generate an offset Hamiltonian.
    _offset = {(0,) * ndim: f_random * charge_op}
    h_offset = add_tb(h_shift, _offset)

    # Compute f_offset for the offset Hamiltonian.
    f_offset = fermi_level(h_offset, charge_op, target_charge, kT, nk, ndim)

    assert np.allclose(f_random, f_offset, kT / 2, kT / 2)


# %%
@pytest.mark.parametrize("seed", range(repeat_number))
def test_minimizer_consistency(seed):
    np.random.seed(seed)
    ndim = np.random.randint(1, 3)
    ndof = np.random.randint(1, 8)
    cutoff = np.random.randint(1, 5)
    nk = np.random.randint(10, 51)
    target_charge = np.random.uniform(0, ndof)
    kT = np.random.uniform(0, 1e-2)
    f_random = np.random.uniform(-3, 3)
    charge_op = np.eye(ndof)
    minimizer_offset(cutoff, ndim, ndof, target_charge, charge_op, nk, kT, f_random)
