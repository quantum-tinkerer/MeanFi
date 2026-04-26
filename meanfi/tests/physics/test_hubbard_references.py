import pytest

from meanfi import (
    AdaptiveQuadrature,
    AndersonMixing,
    Model,
    density_matrix,
    solver,
)
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
    integration = AdaptiveQuadrature(density_matrix_tol=1e-6)
    scf_tol = 1e-5

    model = Model(h_0, h_int, filling=2.0, kT=kT)
    result = solver(
        model,
        antiferromagnetic_guess(0.5 * delta_ref, 1),
        integration=integration,
        scf=AndersonMixing(M=0, line_search="wolfe", max_iterations=80),
        scf_tol=scf_tol,
        filling_tol=1e-6,
    )
    density_result = density_matrix(
        model.hamiltonian_from_meanfield(result.mf),
        filling=2.0,
        kT=kT,
        keys=[(0,)],
        integration=integration,
        filling_tol=1e-6,
    )

    assert result.info.residual_norm <= scf_tol
    assert abs(density_result.filling - model.filling) <= 1e-6
    assert abs(staggered_magnetization(density_result.density_matrix[(0,)]) - m_ref) < 5e-4


def test_solver_matches_antiferromagnetic_gap_equation_in_2d():
    U = 4.0
    kT = 0.3
    h_0, h_int = bipartite_hubbard_2d(U)
    delta_ref = solve_antiferromagnetic_gap(h_0, U=U, kT=kT, ndim=2, nk=140)
    m_ref = 2.0 * delta_ref / U
    integration = AdaptiveQuadrature(density_matrix_tol=1e-5)
    scf_tol = 5e-5

    model = Model(h_0, h_int, filling=2.0, kT=kT)
    result = solver(
        model,
        antiferromagnetic_guess(0.5 * delta_ref, 2),
        integration=integration,
        scf=AndersonMixing(M=0, line_search="wolfe", max_iterations=40),
        scf_tol=scf_tol,
        filling_tol=1e-5,
    )
    density_result = density_matrix(
        model.hamiltonian_from_meanfield(result.mf),
        filling=2.0,
        kT=kT,
        keys=[(0, 0)],
        integration=integration,
        filling_tol=1e-5,
    )

    assert result.info.residual_norm <= scf_tol
    assert abs(density_result.filling - model.filling) <= 1e-5
    assert (
        abs(staggered_magnetization(density_result.density_matrix[(0, 0)]) - m_ref)
        < 2e-3
    )
