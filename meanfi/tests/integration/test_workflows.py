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
from meanfi.tests.fixtures.models import spinful_chain


pytestmark = pytest.mark.integration


def test_graphene_kwant_end_to_end_regression():
    pytest.importorskip("kwant")
    from meanfi.interop import kwant as utils
    from meanfi.tests.fixtures import kwant_examples

    graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
    h_0 = utils.builder_to_tb(graphene_builder)
    h_int = utils.builder_to_tb(int_builder, params={"U": 1.0, "V": 0.0})
    model = Model(h_0, h_int, filling=2.0, kT=0.05)
    guess = model.random_meanfield(rng=0)
    result = solver(
        model,
        guess,
        scf=AndersonMixing(history_size=0, max_iterations=40),
        scf_tol=5e-4,
    )
    density_result = density_matrix(
        add_tb(h_0, result.mean_field),
        filling=2.0,
        kT=model.kT,
        keys=list(h_int),
    )

    assert result.errors.scf_residual <= 2.0 * 5e-4
    assert density_result.errors.filling_residual <= 1e-4
    for key, matrix in result.mean_field.items():
        opposite = tuple(-np.array(key))
        assert np.allclose(matrix, result.mean_field[opposite].conj().T)
        assert np.all(np.isfinite(density_result.to_tb()[key]))


def test_solver_supports_anderson_mixing():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(h_0, h_int, filling=1.0, kT=0.2)

    result = solver(
        model,
        guess,
        scf=AndersonMixing(history_size=0, line_search="wolfe", max_iterations=8),
        scf_tol=1e-8,
    )

    assert result.history
    assert result.errors.scf_residual <= 1e-8
    assert np.allclose(
        result.mean_field[(0,)],
        np.zeros((2, 2)),
        atol=1e-6,
    )


def test_anderson_adapter_records_only_accepted_evaluations_and_forwards_options(
    monkeypatch,
):
    from types import SimpleNamespace
    import meanfi.scf.fixed_point as fixed_point

    calls, evaluated, accepted = {}, [], []

    def evaluate(x):
        x = np.array(x, copy=True)
        evaluated.append(x)
        return SimpleNamespace(
            input_state=SimpleNamespace(values=x),
            residual=x + 0.5,
            residual_norm=float(np.max(abs(x + 0.5))),
        )

    def fake_anderson(func, x0, **kwargs):
        calls.update(kwargs)
        func(x0)
        func(np.array([99.0]))  # Rejected line-search trial.
        residual = func(np.array([1.0]))
        kwargs["callback"](np.array([1.0]), residual)
        return np.array([1.0])

    monkeypatch.setattr(fixed_point, "anderson", fake_anderson)
    fixed_point.iterate_anderson(
        evaluate,
        np.array([0.0]),
        scf=AndersonMixing(
            max_iterations=7,
            alpha=0.3,
            regularization=0.2,
            history_size=4,
            line_search=None,
        ),
        scf_tol=1e-9,
        accept=accepted.append,
    )
    assert [float(x[0]) for x in evaluated] == [0.0, 99.0, 1.0]
    assert [float(item.input_state.values[0]) for item in accepted] == [0.0, 1.0]
    assert calls["alpha"] == 0.3
    assert calls["w0"] == 0.2
    assert calls["M"] == 4
    assert calls["line_search"] is None
    assert calls["maxiter"] == 7
    assert calls["f_tol"] == 1e-9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"alpha": np.inf}, "alpha must be positive"),
        ({"history_size": 0.5}, "history_size must be an integer"),
        ({"max_iterations": True}, "max_iterations must be an integer"),
        ({"regularization": -1.0}, "regularization must be finite and non-negative"),
        ({"line_search": "bad"}, "line_search must be"),
    ],
)
def test_anderson_mixing_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AndersonMixing(**kwargs)


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

    assert abs(result.mu) < 1e-3
    assert np.allclose(
        result.mean_field[(0,)],
        np.zeros((2, 2)),
        atol=1e-3,
    )


def test_default_mixer_handles_reference_restart_and_spatial_symmetry():
    from meanfi import NoConvergence, UniformGrid, SpatialSymmetry

    h = {
        (0,): np.array([[0.15, 0.08j], [-0.08j, -0.1]]),
        (1,): np.diag([-0.7, -0.5]),
        (-1,): np.diag([-0.7, -0.5]),
    }
    interaction = {(0,): np.array([[0.0, 0.25], [0.25, 0.0]])}
    model = Model(h, interaction, filling=0.8, kT=0.2)
    reference = density_matrix(model, tol=1e-6)
    referenced = Model(h, interaction, filling=0.8, kT=0.2, reference=reference)
    swap = SpatialSymmetry(np.eye(1, dtype=int), {(0,): np.array([[0, 1], [1, 0]])})
    symmetric = Model(
        {(0,): np.zeros((2, 2)), (1,): -np.eye(2), (-1,): -np.eye(2)},
        interaction,
        filling=0.8,
        kT=0.2,
        spatial_symmetries=(swap,),
    )
    with pytest.raises(NoConvergence) as caught:
        solver(
            model,
            model.random_meanfield(rng=12, scale=0.03),
            integration=UniformGrid(nk=64),
            scf=LinearMixing(alpha=0.1, max_iterations=1),
            scf_tol=1e-14,
        )
    for problem, guess in (
        (referenced, referenced.random_meanfield(rng=1, scale=0.01)),
        (model, caught.value.result.mean_field),
        (symmetric, symmetric.random_meanfield(rng=2, scale=0.01)),
    ):
        default = solver(problem, guess, tol=1e-5)
        linear = solver(problem, guess, scf=LinearMixing(), tol=1e-5)
        assert default.converged and default.errors.scf_residual <= 1e-5
        assert default.mu == pytest.approx(linear.mu, abs=1e-5)
        for key in default.mean_field:
            np.testing.assert_allclose(
                default.mean_field[key], linear.mean_field[key], atol=1e-5
            )


def test_scf_residual_uses_complex_density_entry_magnitudes():
    import meanfi as mf
    from meanfi.scf.problem import SCFEvaluation
    from meanfi.space.state import ActiveDensityState

    model = mf.Model(
        {(): np.diag([-0.3, 0.4])},
        {(): np.array([[0.0, 1.0], [1.0, 0.0]])},
        1,
        kT=0.2,
    )
    density = mf.density_matrix(model)
    difference = {(): np.array([[0, 0.8e-6 * (1 + 1j)], [0.8e-6 * (1 - 1j), 0]])}
    params = model._space.params_from_density(difference)
    evaluation = SCFEvaluation(
        density,
        ActiveDensityState(model._space, params),
        ActiveDensityState(model._space, np.zeros_like(params)),
    )
    assert np.max(abs(params)) == 0.8e-6
    assert evaluation.residual_norm == pytest.approx(np.sqrt(2) * 0.8e-6)
    assert evaluation.residual_norm > 1e-6
