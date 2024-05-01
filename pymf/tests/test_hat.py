# %%
import numpy as np
from pymf.solvers import solver
from pymf.tb import utils
from pymf.model import Model
from pymf.tb.tb import add_tb, scale_tb
from pymf import mf
from pymf import observables
import pytest


# %%
def total_energy(ham_tb, rho_tb):
    return np.real(observables.expectation_value(rho_tb, ham_tb))


# %%
U0 = 1
filling = 2
nk = 100
repeat_number = 10

hopp = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
h_0 = {(0,): hopp + hopp.T.conj(), (1,): hopp, (-1,): hopp.T.conj()}
h_int_U0 = {
    (0,): U0 * np.kron(np.eye(2), np.ones((2, 2))),
}


# %%
@np.vectorize
def mf_rescaled(alpha, mf0):
    hamiltonian = add_tb(h_0, scale_tb(mf0, alpha))
    rho, _ = mf.density_matrix(hamiltonian, filling=filling, nk=nk)
    hamiltonian = add_tb(h_0, scale_tb(mf0, np.sign(alpha)))
    return total_energy(hamiltonian, rho)

@pytest.mark.parametrize("seed", range(repeat_number))
def test_mexican_hat(seed):
    np.random.seed(seed)
    guess = utils.generate_guess(frozenset(h_int_U0), len(h_int_U0[(0,)]))
    _model = Model(h_0, h_int_U0, filling=filling)
    mf_sol_groundstate = solver(
        _model, mf_guess=guess, nk=nk, optimizer_kwargs={"M": 0}
    )

    alphas = np.random.uniform(0, 50, 100)
    alphas = np.where(alphas == 1, 0, alphas)
    assert np.all(
        mf_rescaled(alphas, mf0=mf_sol_groundstate)
        > mf_rescaled(np.array([1]), mf0=mf_sol_groundstate)
    )
