import numpy as np
import pytest

from meanfi import (
    guess_tb,
    add_tb,
    density_matrix,
)
from meanfi.tb.utils import generate_tb_keys


repeat_number = 5
ndim = 1
ndof = 4
cutoff = 2
nk = 500
filling = ndof / 2
kT = 1e-4


@pytest.mark.parametrize("seed", range(repeat_number))
def test_minimizer_consistency(seed):
    np.random.seed(seed)
    keys = generate_tb_keys(cutoff, ndim)
    h_0 = guess_tb(keys, ndof)

    f_level = density_matrix(h_0, filling, nk, kT)[1]
    
    # Normalize the Hamiltonian.
    _norm  = {(0,)*ndim: -f_level * np.eye(ndof)}
    h_norm = add_tb(h_0, _norm)

    # Generate an offset Hamiltonian.
    f_random = np.random.uniform(-2, 2)
    _offset  = {(0,)*ndim: f_random * np.eye(ndof)}
    h_offset = add_tb(h_norm, _offset)

    # Compute f_offset for the offset Hamiltonian.
    f_offset = density_matrix(h_offset, filling, nk, kT)[1]

    assert np.allclose(f_random, f_offset, kT/2, kT/2)