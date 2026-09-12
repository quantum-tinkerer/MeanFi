from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from meanfi.density.problem import build_normal_problem

from meanfi import (
    ErrorTolerances,
    ErrorValues,
    PeriodicGrid,
    default_solver_tolerances,
    density_matrix,
)


pytestmark = pytest.mark.integration


def _two_level_hamiltonian():
    return {(): np.diag([-1.0, 1.0])}


def test_default_solver_tolerances_define_the_public_error_hierarchy():
    tolerances = default_solver_tolerances(1e-3)

    assert tolerances == ErrorTolerances(
        scf_residual=1e-3,
        density_matrix_integration=2e-4,
        filling_residual=1e-4,
        charge_integration=2e-4,
    )
    with pytest.raises(FrozenInstanceError):
        tolerances.scf_residual = 1e-2


def test_custom_tolerance_function_controls_density_calculation_and_result():
    def precise_density(tol):
        return replace(
            default_solver_tolerances(tol),
            density_matrix_integration=tol / 100,
        )

    result = density_matrix(
        _two_level_hamiltonian(),
        filling=1.0,
        kT=0.2,
        keys=[()],
        tol=1e-4,
        tolerance_policy=precise_density,
    )

    requested = precise_density(1e-4)
    assert result.errors.scf_residual is None
    assert result.errors.density_matrix_integration == pytest.approx(0.0)
    assert result.errors.filling_residual <= requested.filling_residual
    assert result.errors.charge_integration <= requested.charge_integration


def test_explicit_integration_tolerances_are_effective_internal_requests():
    problem = build_normal_problem(
        _two_level_hamiltonian(),
        kT=0.2,
        keys=[()],
        integration=PeriodicGrid(
            density_matrix_tol=5e-7,
            charge_tol=2e-7,
        ),
        tolerances=default_solver_tolerances(1e-4),
    )

    assert problem.tolerances == ErrorTolerances(
        scf_residual=1e-4,
        density_matrix_integration=5e-7,
        filling_residual=1e-5,
        charge_integration=2e-7,
    )


def test_unavailable_periodic_grid_estimators_are_none():
    result = density_matrix(
        _two_level_hamiltonian(),
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=PeriodicGrid(nk=8),
        tol=1e-3,
    )

    assert result.errors == ErrorValues(
        filling_residual=result.errors.filling_residual,
    )
    assert result.errors.scf_residual is None
    assert result.errors.density_matrix_integration is None
    assert result.errors.charge_integration is None
