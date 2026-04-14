import numpy as np

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

    model = Model(h_0, h_int, filling=filling, kT=kT, charge_tol=1e-8, density_atol=1e-8)
    solution, info = solver(model, guess, mu_guess=0.0, return_info=True)
    _, _, mu_expected, density_info = density_matrix(
        h_0,
        filling=filling,
        kT=kT,
        keys=[(0,)],
        charge_tol=1e-8,
        density_atol=1e-8,
    )

    assert np.allclose(solution[(0,)], -mu_expected * np.eye(2), atol=1e-6)
    assert info.iterations >= 1
    assert abs(info.mu - density_info.mu) < 1e-6


def test_solver_supports_linear_mixing_fallback():
    h_0 = _spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}

    model = Model(h_0, h_int, filling=1.0, kT=0.2, scf_tol=1e-8)
    solution, info = solver(
        model,
        guess,
        mixing="linear",
        mixing_kwargs={"alpha": 0.5},
        max_scf_steps=8,
        return_info=True,
    )

    assert info.mixing == "linear"
    assert info.residual_norm <= model.scf_tol
    assert np.allclose(solution[(0,)], -info.mu * np.eye(2), atol=1e-6)
