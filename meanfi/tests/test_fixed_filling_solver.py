import numpy as np
from scipy.optimize import anderson
from types import SimpleNamespace

from meanfi import Model, density_matrix, solver


def _spinful_chain():
    hopping = -np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


def test_solver_matches_fixed_filling_solution_for_zero_interaction():
    h_0 = _spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    filling = 0.7
    kT = 0.15

    model = Model(
        h_0,
        h_int,
        filling=filling,
        kT=kT,
        charge_tol=1e-6,
        density_atol=1e-6,
        scf_tol=1e-6,
    )
    solution, info = solver(model, guess, mu_guess=0.0, return_info=True)
    _, _, mu_expected, density_info = density_matrix(
        h_0,
        filling=filling,
        kT=kT,
        keys=[(0,)],
    )

    assert np.allclose(solution[(0,)], -mu_expected * np.eye(2), atol=1e-6)
    assert info.iterations >= 1
    assert abs(info.mu - density_info.mu) < 1e-6
    assert info.optimizer == "linear_mixing"


def test_solver_supports_explicit_anderson_optimizer():
    h_0 = _spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}

    model = Model(h_0, h_int, filling=1.0, kT=0.2, scf_tol=1e-8)
    solution, info = solver(
        model,
        guess,
        optimizer=anderson,
        optimizer_kwargs={
            "M": 0,
            "line_search": "wolfe",
            "maxiter": 8,
            "f_tol": model.scf_tol,
        },
        max_scf_steps=8,
        return_info=True,
    )

    assert info.optimizer == "anderson"
    assert info.residual_norm <= model.scf_tol
    assert np.allclose(solution[(0,)], -info.mu * np.eye(2), atol=1e-6)


def test_solver_info_residual_norm_uses_max_norm_and_is_not_extensive(monkeypatch):
    import meanfi.solvers as solvers

    def fake_tb_to_rparams(tb):
        return np.asarray(tb[(0,)], dtype=float)

    def fake_rparams_to_tb(params, keys, ndof):
        del keys, ndof
        return {(0,): np.asarray(params, dtype=float)}

    def fake_meanfield(rho, h_int):
        del rho, h_int
        return {}

    def fake_info():
        return SimpleNamespace(
            charge_integration_calls=0,
            density_integration_calls=1,
            n_kernel_evals=0,
            n_evaluator_evals=0,
            error_estimate_available=True,
        )

    class FakeModel:
        def __init__(self, step):
            self.step = np.asarray(step, dtype=float)
            self.h_int = {(0,): np.zeros((1, 1))}
            self._ndof = 1
            self._local_key = (0,)
            self.scf_tol = 0.2

        def hamiltonian_from_meanfield(self, mf):
            return mf

        def density_matrix(self, rho, *, keys, mu_guess):
            del keys, mu_guess
            params = np.asarray(rho[(0,)], dtype=float)
            return {(0,): params + self.step}, None, 0.0, fake_info()

    def fake_density_for_hamiltonian(model, hamiltonian, *, keys, mu_guess):
        del hamiltonian, keys, mu_guess
        return {(0,): np.zeros_like(model.step)}, None, 0.0, fake_info()

    monkeypatch.setattr(solvers, "tb_to_rparams", fake_tb_to_rparams)
    monkeypatch.setattr(solvers, "rparams_to_tb", fake_rparams_to_tb)
    monkeypatch.setattr(solvers, "meanfield", fake_meanfield)
    monkeypatch.setattr(solvers, "_density_for_hamiltonian", fake_density_for_hamiltonian)

    _, info_short = solvers.solver(
        FakeModel([0.1, -0.02]),
        {(0,): np.zeros((1, 1))},
        return_info=True,
        max_scf_steps=1,
    )
    _, info_long = solvers.solver(
        FakeModel([0.1, -0.02, 0.1, -0.02, 0.1, -0.02]),
        {(0,): np.zeros((1, 1))},
        return_info=True,
        max_scf_steps=1,
    )

    assert np.isclose(info_short.residual_norm, 0.1)
    assert np.isclose(info_long.residual_norm, 0.1)
