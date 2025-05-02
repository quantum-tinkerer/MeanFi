# %%
import numpy as np
import pytest

from meanfi import (
    Model,
    solver,
    generate_tb_vals,
    scale_tb,
    add_tb,
    expectation_value,
    density_matrix,
)

from meanfi.tb.utils import generate_tb_keys


# %%
def total_energy(ham_tb, rho_tb):
    return np.real(expectation_value(rho_tb, ham_tb))


# %%
U0 = 1
target_charge = 2
nk = 10
repeat_number = 3
ndof = 4
cutoff = 1
charge_op = np.eye(ndof)
kT = 0


# %%
@np.vectorize
def mf_rescaled(alpha, mf0, h0):
    hamiltonian = add_tb(h0, scale_tb(mf0, alpha))
    rho, _ = density_matrix(hamiltonian, charge_op, target_charge, kT, nk=nk)
    hamiltonian = add_tb(h0, scale_tb(mf0, np.sign(alpha)))
    return total_energy(hamiltonian, rho)


@pytest.mark.parametrize("seed", range(repeat_number))
def test_mexican_hat(seed):
    np.random.seed(seed)
    h0s = []
    h_ints = []
    for ndim in np.arange(4):
        keys = generate_tb_keys(cutoff, ndim)
        h0s.append(generate_tb_vals(keys, ndof))
        h_int = generate_tb_vals(keys, ndof)
        h_int[keys[len(keys) // 2]] += U0
        h_ints.append(h_int)

    for h0, h_int in zip(h0s, h_ints):
        guess = generate_tb_vals(frozenset(h_int), ndof)
        _model = Model(h0, h_int, target_charge, charge_op, kT)
        mf_sol_groundstate = solver(
            _model, mf_guess=guess, nk=nk, optimizer_kwargs={"M": 0}
        )

        alphas = np.random.uniform(0, 50, 100)
        alphas = np.where(alphas == 1, 0, alphas)
        assert np.all(
            mf_rescaled(alphas, mf0=mf_sol_groundstate, h0=h0)
            > mf_rescaled(np.array([1]), mf0=mf_sol_groundstate, h0=h0)
        )
