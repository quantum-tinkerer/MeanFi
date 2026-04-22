import pytest
from scipy.optimize import anderson

from meanfi import Model, density_matrix, solver
from meanfi.tests.helpers import (
    antiferromagnetic_guess,
    bipartite_hubbard_1d,
    bipartite_hubbard_2d,
    solve_antiferromagnetic_gap,
    staggered_magnetization,
)


pytestmark = pytest.mark.physics


def test_solver_matches_antiferromagnetic_gap_equation_in_1d():
    U = 8.0
    kT = 0.05
    h_0, h_int = bipartite_hubbard_1d(U)
    delta_ref = solve_antiferromagnetic_gap(h_0, U=U, kT=kT, ndim=1, nk=4000)
    m_ref = 2.0 * delta_ref / U

    model = Model(
        h_0,
        h_int,
        filling=2.0,
        kT=kT,
        charge_tol=1e-6,
        density_atol=1e-6,
        scf_tol=1e-5,
    )
    mf_sol, solver_info = solver(
        model,
        antiferromagnetic_guess(0.5 * delta_ref, 1),
        optimizer=anderson,
        optimizer_kwargs={
            "M": 0,
            "line_search": "wolfe",
            "maxiter": 80,
            "f_tol": model.scf_tol,
        },
        max_scf_steps=80,
        return_info=True,
    )
    rho, _, _, density_info = density_matrix(
        model.hamiltonian_from_meanfield(mf_sol),
        filling=2.0,
        kT=kT,
        keys=[(0,)],
        charge_tol=1e-6,
        density_atol=1e-6,
    )

    assert solver_info.residual_norm <= model.scf_tol
    assert abs(density_info.charge - model.filling) <= model.charge_tol
    assert abs(staggered_magnetization(rho[(0,)]) - m_ref) < 5e-4


def test_solver_matches_antiferromagnetic_gap_equation_in_2d():
    U = 4.0
    kT = 0.3
    h_0, h_int = bipartite_hubbard_2d(U)
    delta_ref = solve_antiferromagnetic_gap(h_0, U=U, kT=kT, ndim=2, nk=140)
    m_ref = 2.0 * delta_ref / U

    model = Model(
        h_0,
        h_int,
        filling=2.0,
        kT=kT,
        charge_tol=1e-5,
        density_atol=1e-5,
        scf_tol=5e-5,
    )
    mf_sol, solver_info = solver(
        model,
        antiferromagnetic_guess(0.5 * delta_ref, 2),
        optimizer=anderson,
        optimizer_kwargs={
            "M": 0,
            "line_search": "wolfe",
            "maxiter": 40,
            "f_tol": model.scf_tol,
        },
        max_scf_steps=40,
        return_info=True,
    )
    rho, _, _, density_info = density_matrix(
        model.hamiltonian_from_meanfield(mf_sol),
        filling=2.0,
        kT=kT,
        keys=[(0, 0)],
        charge_tol=1e-5,
        density_atol=1e-5,
    )

    assert solver_info.residual_norm <= model.scf_tol
    assert abs(density_info.charge - model.filling) <= model.charge_tol
    assert abs(staggered_magnetization(rho[(0, 0)]) - m_ref) < 2e-3
