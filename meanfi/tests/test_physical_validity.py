import numpy as np
from scipy.optimize import brentq
from scipy.optimize import anderson

from meanfi import Model, density_matrix, density_matrix_at_mu, fermi_dirac, solver, tb_to_kfunc
from meanfi.tb.tb import add_tb


def _onsite_hubbard_interaction(U: float, ndim: int):
    return {(0,) * ndim: U * np.kron(np.eye(2), np.ones((2, 2)))}


def _bipartite_hubbard_1d(U: float):
    hop = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {(0,): hop + hop.T.conj(), (1,): hop, (-1,): hop.T.conj()}
    return h_0, _onsite_hubbard_interaction(U, 1)


def _bipartite_hubbard_2d(U: float):
    hop = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {
        (0, 0): hop + hop.T.conj(),
        (1, 0): hop,
        (-1, 0): hop.T.conj(),
        (0, 1): hop,
        (0, -1): hop.T.conj(),
        (1, 1): hop,
        (-1, -1): hop.T.conj(),
    }
    return h_0, _onsite_hubbard_interaction(U, 2)


def _antiferromagnetic_guess(delta: float, ndim: int):
    key = (0,) * ndim
    return {key: np.diag([-delta, delta, delta, -delta]).astype(complex)}


def _staggered_magnetization(local_density: np.ndarray) -> float:
    occupations = np.real(np.diag(local_density))
    magnetization = 0.5 * ((occupations[0] - occupations[1]) + (occupations[3] - occupations[2]))
    return float(abs(magnetization))


def _gamma_abs_samples(h_0, ndim: int, nk: int) -> np.ndarray:
    hkfunc = tb_to_kfunc(h_0)
    if ndim == 1:
        ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)[:, None]
    elif ndim == 2:
        k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        ks = np.stack([kx.ravel(), ky.ravel()], axis=-1)
    else:
        raise ValueError("Only 1D and 2D antiferromagnetic references are supported")

    h_k = hkfunc(ks)
    spin_up_block = h_k[:, [0, 2]][:, :, [0, 2]]
    return np.abs(spin_up_block[:, 0, 1])


def _solve_antiferromagnetic_gap(h_0, U: float, kT: float, ndim: int, nk: int) -> float:
    gamma_abs = _gamma_abs_samples(h_0, ndim, nk)

    def residual(delta: float) -> float:
        energy = np.sqrt(gamma_abs**2 + delta**2)
        return 1.0 - 0.5 * U * np.mean(np.tanh(energy / (2.0 * kT)) / energy)

    if residual(1e-12) >= 0.0:
        return 0.0

    upper = max(10.0, 2.0 * U)
    return float(brentq(residual, 1e-12, upper))


def test_density_matrix_at_mu_is_exact_in_zero_dimensional_limit():
    h = {(): np.diag([-1.0, 2.0])}
    mu = 0.3
    kT = 0.4

    rho, error, info = density_matrix_at_mu(h, mu=mu, kT=kT, keys=[()])
    expected = np.diag(fermi_dirac(np.array([-1.0, 2.0]), kT, mu))

    assert np.allclose(rho[()], expected, atol=1e-12)
    assert np.allclose(error[()], np.zeros((2, 2)), atol=0.0)
    assert info.n_kernel_evals == 1
    assert info.n_evaluator_evals == 1


def test_density_matrix_respects_hermiticity_and_charge_sum_rule():
    h_0, _ = _bipartite_hubbard_1d(U=4.0)
    h_trial = add_tb(h_0, _antiferromagnetic_guess(0.7, 1))

    rho, _, _, info = density_matrix(
        h_trial,
        filling=2.0,
        kT=0.1,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-8,
        density_atol=1e-8,
    )

    assert np.allclose(rho[(0,)], rho[(0,)].conj().T, atol=1e-8)
    assert np.allclose(rho[(-1,)], rho[(1,)].conj().T, atol=1e-8)
    assert abs(np.trace(rho[(0,)]).real - 2.0) <= 1e-8
    assert abs(info.charge - 2.0) <= 1e-8


def test_solver_matches_antiferromagnetic_gap_equation_in_1d():
    U = 8.0
    kT = 0.05
    h_0, h_int = _bipartite_hubbard_1d(U)
    delta_ref = _solve_antiferromagnetic_gap(h_0, U=U, kT=kT, ndim=1, nk=4000)
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
        _antiferromagnetic_guess(0.5 * delta_ref, 1),
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
    h_full = model.hamiltonian_from_meanfield(mf_sol)
    rho, _, _, density_info = density_matrix(
        h_full,
        filling=2.0,
        kT=kT,
        keys=[(0,)],
        charge_tol=1e-6,
        density_atol=1e-6,
    )
    m_numeric = _staggered_magnetization(rho[(0,)])

    assert solver_info.residual_norm <= model.scf_tol
    assert abs(density_info.charge - model.filling) <= model.charge_tol
    assert abs(m_numeric - m_ref) < 5e-4


def test_solver_matches_antiferromagnetic_gap_equation_in_2d():
    U = 4.0
    kT = 0.3
    h_0, h_int = _bipartite_hubbard_2d(U)
    delta_ref = _solve_antiferromagnetic_gap(h_0, U=U, kT=kT, ndim=2, nk=140)
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
        _antiferromagnetic_guess(0.5 * delta_ref, 2),
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
    h_full = model.hamiltonian_from_meanfield(mf_sol)
    rho, _, _, density_info = density_matrix(
        h_full,
        filling=2.0,
        kT=kT,
        keys=[(0, 0)],
        charge_tol=1e-5,
        density_atol=1e-5,
    )
    m_numeric = _staggered_magnetization(rho[(0, 0)])

    assert solver_info.residual_norm <= model.scf_tol
    assert abs(density_info.charge - model.filling) <= model.charge_tol
    assert abs(m_numeric - m_ref) < 2e-3
