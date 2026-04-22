import numpy as np
import pytest
from scipy.optimize import anderson

from meanfi import (
    Model,
    add_tb,
    density_matrix,
    guess_tb,
    solver,
    tb_to_kfunc,
    tb_to_tight_binding_model,
)
from meanfi.kwant_helper import kwant_examples, utils
from meanfi.zero_temp import _NATIVE_ZERO_TEMP_AVAILABLE
from meanfi.tests.helpers import qiwuzhang, spinful_chain


pytestmark = pytest.mark.integration
requires_native = pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)


def test_graphene_kwant_end_to_end_regression():
    graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
    h_0 = utils.builder_to_tb(graphene_builder)
    h_int = utils.builder_to_tb(int_builder, {"U": 1.0, "V": 0.0})
    np.random.seed(0)
    guess = guess_tb(frozenset(h_int), len(next(iter(h_0.values()))))

    model = Model(
        h_0,
        h_int,
        filling=2.0,
        kT=0.05,
        charge_tol=1e-6,
        density_atol=1e-6,
        scf_tol=5e-4,
    )
    mf_sol, solver_info = solver(
        model,
        guess,
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
    rho, _error, _mu, density_info = density_matrix(
        add_tb(h_0, mf_sol),
        filling=2.0,
        kT=model.kT,
        keys=list(h_int),
        charge_tol=model.charge_tol,
        density_atol=model.density_atol,
    )

    assert solver_info.residual_norm <= 2.0 * model.scf_tol
    assert abs(density_info.charge - model.filling) <= model.charge_tol
    for key, matrix in mf_sol.items():
        opposite = tuple(-np.array(key))
        assert np.allclose(matrix, mf_sol[opposite].conj().T)
        assert np.all(np.isfinite(rho[key]))


def test_solver_supports_explicit_anderson_optimizer():
    h_0 = spinful_chain()
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


@requires_native
def test_zero_temperature_model_solver_workflow_supports_zero_interaction():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.0,
        charge_tol=1e-3,
        density_atol=1e-3,
        scf_tol=1e-3,
    )

    solution, info = solver(model, guess, return_info=True)

    assert abs(info.mu) < 1e-3
    assert np.allclose(solution[(0,)], -info.mu * np.eye(2), atol=1e-3)


@requires_native
def test_public_native_tight_binding_model_matches_python_kfunc():
    tb = qiwuzhang()
    native_model = tb_to_tight_binding_model(tb)
    hkfunc = tb_to_kfunc(tb)
    points = np.array(
        [
            [0.0, 0.0],
            [0.3, -0.7],
            [-np.pi / 2.0, np.pi / 3.0],
        ],
        dtype=float,
    )

    assert native_model.ndim == 2
    assert native_model.ndof == 2
    assert native_model.nterms == len(tb)
    assert np.allclose(native_model.evaluate_point(points[0]), hkfunc(points[0]))
    assert np.allclose(native_model.evaluate_many(points), hkfunc(points))

