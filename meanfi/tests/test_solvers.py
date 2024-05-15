import numpy as np
import pytest
from scipy.optimize._nonlin import NoConvergence

from meanfi import (
    Model,
    guess_tb,
    add_tb,
    density_matrix,
)
from meanfi.solvers import solver_density, solver_mf
from meanfi.tb.utils import generate_tb_keys
from meanfi.tb.tb import compare_dicts

repeat_number = 5
ndim = 1
ndof = 4
cutoff = 2
U0 = 1
nk = 500
filling = ndof / 2


@pytest.mark.parametrize("seed", range(repeat_number))
def test_solver_consistency(seed):
    np.random.seed(seed)
    keys = generate_tb_keys(cutoff, ndim)
    h0 = guess_tb(keys, ndof)
    h_int = guess_tb(keys, ndof)
    h_int[keys[len(keys) // 2]] += U0
    guess = guess_tb(frozenset(h_int), ndof)
    _model = Model(h0, h_int, filling=filling)

    def solve_for_rho(model, solver):
        mf_sol = solver(
            model,
            mf_guess=guess,
            nk=nk,
            optimizer_kwargs={"M": 0, "f_tol": 1e-3, "maxiter": 10000},
        )
        mf_full = add_tb(h0, mf_sol)
        rho, _ = density_matrix(mf_full, filling=filling, nk=nk)
        rho = {key: rho[key] for key in h_int}
        return rho

    try:
        rho_dens = solve_for_rho(_model, solver_density)
        rho_mf = solve_for_rho(_model, solver_mf)
    except NoConvergence:
        guess = guess_tb(frozenset(h_int), ndof)
        rho_dens = solve_for_rho(_model, solver_density)
        rho_mf = solve_for_rho(_model, solver_mf)

    compare_dicts(rho_dens, rho_mf, atol=1e-2)
