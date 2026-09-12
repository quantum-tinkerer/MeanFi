import numpy as np
import pytest

from meanfi import (
    PeriodicGrid,
    LinearMixing,
    Model,
    SCFResult,
    SolverFailure,
    solver,
)
from meanfi.tests.fixtures.models import spinful_chain


pytestmark = pytest.mark.integration


def test_numerical_failure_attaches_last_valid_physical_result(monkeypatch):
    import meanfi.scf.normal as normal_scf

    original = normal_scf._density_update_for_normal_hamiltonian
    calls = 0

    def fail_after_initial_density(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError("synthetic density failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        normal_scf,
        "_density_update_for_normal_hamiltonian",
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
            integration=PeriodicGrid(density_matrix_tol=1e-4),
            scf=LinearMixing(max_iterations=3),
        )

    assert calls == 2
    assert isinstance(exc_info.value.result, SCFResult)
    assert exc_info.value.result.converged is False
    assert np.isfinite(exc_info.value.result.mu)
    assert np.isfinite(exc_info.value.result.filling)
    assert isinstance(exc_info.value.__cause__, ArithmeticError)
