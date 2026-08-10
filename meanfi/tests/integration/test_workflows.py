import numpy as np
import pytest

from meanfi import (
    AndersonMixing,
    LinearMixing,
    Model,
    add_tb,
    density_matrix,
    solver,
)
from meanfi.interop import kwant as utils
from meanfi.tests.fixtures import kwant_examples
from meanfi.density.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.tests.fixtures.models import spinful_chain


pytestmark = pytest.mark.integration
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def test_graphene_kwant_end_to_end_regression():
    graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
    h_0 = utils.builder_to_tb(graphene_builder)
    h_int = utils.builder_to_tb(int_builder, {"U": 1.0, "V": 0.0})
    model = Model(h_0, h_int, filling=2.0, kT=0.05)
    guess = model.random_meanfield(rng=0)
    result = solver(
        model,
        guess,
        scf=AndersonMixing(M=0, max_iterations=40),
        scf_tol=5e-4,
    )
    density_result = density_matrix(
        add_tb(h_0, result.mf),
        filling=2.0,
        kT=model.kT,
        keys=list(h_int),
    )

    assert result.info.residual_norm <= 2.0 * 5e-4
    assert (
        density_result.errors.filling_residual
        <= density_result.tolerances.filling_residual
    )
    for key, matrix in result.mf.items():
        opposite = tuple(-np.array(key))
        assert np.allclose(matrix, result.mf[opposite].conj().T)
        assert np.all(np.isfinite(density_result.density_matrix[key]))


def test_solver_supports_anderson_mixing():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(h_0, h_int, filling=1.0, kT=0.2)

    result = solver(
        model,
        guess,
        scf=AndersonMixing(M=0, line_search="wolfe", max_iterations=8),
        scf_tol=1e-8,
    )

    assert result.info.method == "anderson_mixing"
    assert result.info.residual_norm <= 1e-8
    assert np.allclose(
        result.mf[(0,)],
        -result.density_matrix_result.mu * np.eye(2),
        atol=1e-6,
    )


def test_anderson_mixing_forwards_scipy_options(monkeypatch):
    import meanfi.scf.fixed_point as fixed_point

    calls = {}

    def fake_anderson(func, x0, **kwargs):
        calls.update(kwargs)
        return np.asarray(x0, dtype=float)

    monkeypatch.setattr(fixed_point, "anderson", fake_anderson)

    result = fixed_point.solve_fixed_point(
        lambda x: np.zeros_like(x),
        np.array([1.0]),
        scf=AndersonMixing(
            max_iterations=7,
            alpha=0.3,
            w0=0.2,
            M=4,
            f_rtol=1e-6,
            x_tol=1e-7,
            x_rtol=1e-8,
            line_search=None,
        ),
        scf_tol=1e-9,
        on_iteration=lambda *_args: None,
    )

    assert np.allclose(result, [1.0])
    assert calls["alpha"] == 0.3
    assert calls["w0"] == 0.2
    assert calls["M"] == 4
    assert calls["f_rtol"] == 1e-6
    assert calls["x_tol"] == 1e-7
    assert calls["x_rtol"] == 1e-8
    assert calls["line_search"] is None
    assert calls["maxiter"] == 7
    assert calls["f_tol"] == 1e-9


def test_anderson_mixing_reports_only_accepted_iterations(monkeypatch):
    import meanfi.scf.fixed_point as fixed_point

    events = []

    def fake_anderson(func, x0, **kwargs):
        func(np.asarray(x0, dtype=float))
        func(np.array([99.0]))
        kwargs["callback"](np.array([1.0]), np.array([0.25]))
        return np.array([1.0])

    def on_iteration(iteration, residual_norm, params, residual):
        events.append(
            (
                iteration,
                residual_norm,
                np.asarray(params, dtype=float).copy(),
                np.asarray(residual, dtype=float).copy(),
            )
        )

    monkeypatch.setattr(fixed_point, "anderson", fake_anderson)

    result = fixed_point.solve_fixed_point(
        lambda x: np.asarray(x, dtype=float) + 0.5,
        np.array([0.0]),
        scf=AndersonMixing(max_iterations=3),
        scf_tol=1e-9,
        on_iteration=on_iteration,
    )

    assert np.allclose(result, [1.0])
    assert len(events) == 2
    assert events[0][0] is None
    assert events[0][1] == 0.5
    assert np.allclose(events[0][2], [0.0])
    assert np.allclose(events[0][3], [0.5])
    assert events[1][0] == 1
    assert events[1][1] == 0.25
    assert np.allclose(events[1][2], [1.0])
    assert np.allclose(events[1][3], [0.25])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"w0": -1.0}, "w0 must be non-negative"),
        ({"f_rtol": 0.0}, "f_rtol must be positive"),
        ({"x_tol": 0.0}, "x_tol must be positive"),
        ({"x_rtol": 0.0}, "x_rtol must be positive"),
        ({"line_search": "bad"}, "line_search must be"),
    ],
)
def test_anderson_mixing_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AndersonMixing(**kwargs)


@requires_ext
def test_zero_temperature_model_solver_workflow_supports_zero_interaction():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(h_0, h_int, filling=1.0)

    result = solver(
        model,
        guess,
        scf=LinearMixing(),
        scf_tol=1e-3,
    )

    assert abs(result.density_matrix_result.mu) < 1e-3
    assert np.allclose(
        result.mf[(0,)],
        -result.density_matrix_result.mu * np.eye(2),
        atol=1e-3,
    )
