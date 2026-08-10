from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from meanfi import (
    AdaptiveQuadrature,
    ErrorTolerances,
    ErrorValues,
    UniformGrid,
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
        density_matrix_integration=1e-4,
        filling_residual=1e-4,
        charge_integration=1e-5,
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

    assert result.tolerances == ErrorTolerances(
        scf_residual=1e-4,
        density_matrix_integration=1e-6,
        filling_residual=1e-5,
        charge_integration=1e-6,
    )
    assert isinstance(result.integration, AdaptiveQuadrature)
    assert result.integration.density_matrix_tol == pytest.approx(1e-6)
    assert result.integration.charge_tol == pytest.approx(1e-6)
    assert result.errors.scf_residual is None
    assert result.errors.density_matrix_integration == pytest.approx(0.0)
    assert result.errors.filling_residual <= result.tolerances.filling_residual
    assert result.errors.charge_integration <= result.tolerances.charge_integration


def test_explicit_integration_tolerances_are_reported_as_effective_requests():
    result = density_matrix(
        _two_level_hamiltonian(),
        filling=1.0,
        kT=0.2,
        keys=[()],
        tol=1e-4,
        integration=AdaptiveQuadrature(
            density_matrix_tol=5e-7,
            charge_tol=2e-7,
        ),
    )

    assert result.tolerances == ErrorTolerances(
        scf_residual=1e-4,
        density_matrix_integration=5e-7,
        filling_residual=1e-5,
        charge_integration=2e-7,
    )


def test_unavailable_uniform_grid_estimators_are_none():
    result = density_matrix(
        _two_level_hamiltonian(),
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=UniformGrid(nk=8),
        tol=1e-3,
    )

    assert result.errors == ErrorValues(
        filling_residual=result.filling_residual,
    )
    assert result.errors.scf_residual is None
    assert result.errors.density_matrix_integration is None
    assert result.errors.charge_integration is None
