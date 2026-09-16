from dataclasses import replace
from meanfi import default_solver_tolerances
import numpy as np
import pytest

from meanfi import (
    UniformGrid,
    LinearMixing,
    Model,
    SCFResult,
    SolverFailure,
    solver,
)
from meanfi.tests.fixtures.models import spinful_chain


pytestmark = pytest.mark.integration


def test_numerical_failure_attaches_last_valid_physical_result(monkeypatch):
    import meanfi.scf.problem as scf_problem

    original = scf_problem.evaluate_density
    calls = 0

    def fail_after_initial_density(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError("synthetic density failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scf_problem,
        "evaluate_density",
        fail_after_initial_density,
    )
    model = Model(
        spinful_chain(),
        {(0,): np.eye(2)},
        filling=1.0,
        kT=0.2,
    )

    with pytest.raises(SolverFailure) as exc_info:
        solver(
            model,
            {(0,): np.zeros((2, 2))},
            integration=UniformGrid(),
            scf=LinearMixing(max_iterations=3),
            tol=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-4,
                charge_integration=1e-4,
            ),
        )

    assert calls == 2
    assert exc_info.value.result.entropy is None
    assert isinstance(exc_info.value.result, SCFResult)
    assert exc_info.value.result.converged is False
    assert np.isfinite(exc_info.value.result.mu)
    assert np.isfinite(exc_info.value.result.filling)
    assert isinstance(exc_info.value.__cause__, ArithmeticError)
