import numpy as np
import pytest

from meanfi import (
    AdaptiveQuadrature,
    LinearMixing,
    Model,
    NoConvergence,
    SCFResult,
    SCFIteration,
    solver,
)
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration


def test_solver_raises_no_convergence_when_scf_budget_is_exhausted():
    model = Model(
        spinful_chain(),
        {(0,): np.eye(2)},
        filling=1.0,
        kT=0.1,
    )

    with pytest.raises(NoConvergence) as exc_info:
        solver(
            model,
            {(0,): 0.2 * np.eye(2)},
            integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
            scf=LinearMixing(max_iterations=1, alpha=0.1),
            scf_tol=1e-30,
        )

    assert exc_info.value.last_iterate.size > 0
    partial = exc_info.value.result
    assert isinstance(partial, SCFResult)
    assert partial.converged is False
    assert partial.history
    assert partial.history[-1].errors.scf_residual == partial.errors.scf_residual
    assert np.isfinite(partial.mu)
    assert np.isfinite(partial.filling)


def test_solver_result_exposes_compact_scf_iteration_history():
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2))},
        filling=1.0,
        kT=0.1,
    )
    result = solver(
        model,
        {(0,): np.zeros((2, 2))},
        integration=AdaptiveQuadrature(density_matrix_tol=1e-5),
        scf=LinearMixing(max_iterations=3),
        scf_tol=1e-5,
    )

    history = result.history
    assert history
    assert all(isinstance(item, SCFIteration) for item in history)
    assert np.isclose(history[-1].errors.scf_residual, result.errors.scf_residual)
    assert [item.step for item in history] == list(range(1, len(history) + 1))
    assert all(np.isfinite(item.mu) for item in history)
    assert all(np.isfinite(item.filling) for item in history)
    assert result.converged is True


def test_solver_verbose_prints_scf_progress(capsys):
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2))},
        filling=1.0,
        kT=0.1,
    )
    result = solver(
        model,
        {(0,): np.zeros((2, 2))},
        integration=AdaptiveQuadrature(density_matrix_tol=1e-5),
        scf=LinearMixing(max_iterations=3),
        scf_tol=1e-5,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert result.history
    assert "scf step=1" in output
    assert "residual=" in output
    assert "mu=" in output
    assert "filling=" in output
    assert "charge_error=" in output
