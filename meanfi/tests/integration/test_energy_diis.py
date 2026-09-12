from __future__ import annotations

import numpy as np
import pytest
from fermisimplex import SpectralMesh

from meanfi import (
    PeriodicGrid,
    AdaptiveSimplex,
    EnergyDIIS,
    ErrorTolerances,
    Model,
    solver,
)
from meanfi.scf.ediis import EDIISPoint, ediis_coefficients


pytestmark = pytest.mark.integration


def test_ediis_minimizes_exact_quadratic_energy_on_convex_hull():
    history = [
        EDIISPoint(np.array([0.0]), 1.0, 1.0),
        EDIISPoint(np.array([2.0]), -3.0, 1.0),
    ]

    coefficients = ediis_coefficients(
        history,
        interaction_energy=lambda params: float(params[0] ** 2),
        interaction_gradient=lambda params, direction: float(
            2.0 * params[0] * direction[0]
        ),
    )

    assert coefficients == pytest.approx([0.5, 0.5], abs=1e-8)


def _zero_dimensional_model(*, kT: float = 0.0) -> Model:
    return Model(
        {(): np.diag([-1.0, 1.0]).astype(complex)},
        {(): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)},
        filling=1.0,
        kT=kT,
    )


def test_default_zero_temperature_adaptive_solver_reports_energy():
    result = solver(
        _zero_dimensional_model(),
        {(): np.zeros((2, 2), dtype=complex)},
        integration=AdaptiveSimplex(density_matrix_tol=1e-6),
        scf_tol=1e-7,
    )

    assert result.converged is True
    assert result.history
    assert not hasattr(result, "accuracy")
    assert result.total_energy == pytest.approx(-1.0)


def test_explicit_energy_diis_uses_requested_tolerances_from_first_iteration():
    result = solver(
        _zero_dimensional_model(),
        {(): np.zeros((2, 2), dtype=complex)},
        integration=AdaptiveSimplex(density_matrix_tol=1e-6),
        scf=EnergyDIIS(),
        scf_tol=1e-7,
    )

    assert len(result.history) == 1
    assert result.total_energy == pytest.approx(-1.0)


def test_energy_diis_evaluates_the_tolerance_policy_once_for_the_solve():
    calls = []

    def custom_tolerances(tol):
        calls.append(tol)
        return ErrorTolerances(
            scf_residual=tol,
            density_matrix_integration=tol / 20,
            filling_residual=tol / 5,
            charge_integration=tol / 50,
        )

    result = solver(
        _zero_dimensional_model(),
        {(): np.zeros((2, 2), dtype=complex)},
        scf=EnergyDIIS(),
        tol=1e-6,
        tolerance_policy=custom_tolerances,
    )

    requested = ErrorTolerances(
        scf_residual=1e-6,
        density_matrix_integration=5e-8,
        filling_residual=2e-7,
        charge_integration=2e-8,
    )
    assert calls == [1e-6]
    assert result.errors.scf_residual <= requested.scf_residual
    assert result.errors.density_matrix_integration <= (
        requested.density_matrix_integration
    )
    assert result.errors.filling_residual <= requested.filling_residual


def test_other_capabilities_keep_anderson():
    result = solver(
        _zero_dimensional_model(kT=0.2),
        {(): np.zeros((2, 2), dtype=complex)},
        integration=PeriodicGrid(density_matrix_tol=1e-8),
        scf_tol=1e-7,
    )

    assert result.history
    assert result.total_energy is None


def test_energy_diis_rejects_unsupported_integration():
    with pytest.raises(ValueError, match="normal-state zero-temperature"):
        solver(
            _zero_dimensional_model(kT=0.2),
            {(): np.zeros((2, 2), dtype=complex)},
            integration=PeriodicGrid(),
            scf=EnergyDIIS(),
        )


@pytest.mark.skipif(
    not hasattr(SpectralMesh, "occupied_weights"),
    reason="installed FermiSimplex does not yet provide occupied weights",
)
def test_energy_diis_uses_cached_occupied_weights_for_periodic_model():
    hopping = np.diag([-0.5, -0.5]).astype(complex)
    model = Model(
        {
            (0,): np.zeros((2, 2), dtype=complex),
            (1,): hopping,
            (-1,): hopping,
        },
        {(0,): np.array([[0.0, 2.0], [2.0, 0.0]], dtype=complex)},
        filling=1.0,
        kT=0.0,
    )

    result = solver(
        model,
        {(0,): np.diag([0.2, -0.2]).astype(complex)},
        integration=AdaptiveSimplex(
            density_matrix_tol=2e-3,
            max_refinements=500,
        ),
        scf=EnergyDIIS(),
        scf_tol=3e-3,
    )

    assert np.isfinite(result.total_energy)
    assert all(np.isfinite(item.total_energy) for item in result.history)
    assert result.errors.scf_residual <= 3e-3
