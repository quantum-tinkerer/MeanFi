import numpy as np
import pytest
from scipy.optimize._nonlin import NoConvergence

from meanfi import (
    Model,
    generate_tb_vals,
    add_tb,
    density_matrix,
)
from meanfi.solvers import solver_density, solver_mf  # , solver_density_symmetric
from meanfi.tb.utils import generate_tb_keys
from meanfi.tb.tb import compare_dicts

repeat_number = 5
ndim = 1
ndof = 4
cutoff = 2
U0 = 1
nk = 500
target_charge = ndof / 2


@pytest.mark.parametrize("seed", range(repeat_number))
def test_solver_consistency(seed):
    np.random.seed(seed)
    keys = generate_tb_keys(cutoff, ndim)
    h0 = generate_tb_vals(keys, ndof)
    h_int = generate_tb_vals(keys, ndof)
    h_int[keys[len(keys) // 2]] += U0
    guess = generate_tb_vals(frozenset(h_int), ndof)
    _model = Model(h0, h_int, target_charge)

    def solve_for_rho(model, solver):
        mf_sol = solver(
            model,
            mf_guess=guess,
            nk=nk,
            optimizer_kwargs={"M": 0, "f_tol": 1e-3, "maxiter": 10000},
        )
        mf_full = add_tb(h0, mf_sol)
        rho, _ = density_matrix(mf_full, _model.charge_op, target_charge, model.kT, nk)
        rho = {key: rho[key] for key in h_int}
        return rho

    try:
        rho_dens = solve_for_rho(_model, solver_density)
        rho_mf = solve_for_rho(_model, solver_mf)
    except NoConvergence:
        guess = generate_tb_vals(frozenset(h_int), ndof)
        rho_dens = solve_for_rho(_model, solver_density)
        rho_mf = solve_for_rho(_model, solver_mf)

    compare_dicts(rho_dens, rho_mf, atol=1e-2)
