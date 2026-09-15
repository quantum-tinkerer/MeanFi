from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from meanfi.density.problem import build_density_problem

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
        band_energy_integration=2e-4,
        entropy_integration=2e-4,
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
    problem = build_density_problem(
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
        band_energy_integration=2e-5,
        entropy_integration=2e-5,
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


@pytest.mark.parametrize("energy_scale", [1.0, 20.0])
def test_energy_and_entropy_targets_are_independent_and_have_physical_units(
    energy_scale,
):
    from scipy.special import entr, expit
    from meanfi import density_matrix_at_mu

    h = {(0,): np.array([[0.13]]), (1,): np.array([[0.5]]), (-1,): np.array([[0.5]])}
    h = {key: energy_scale * block for key, block in h.items()}
    mu, kT = energy_scale * 0.27, energy_scale * 0.037
    energy_tol, entropy_tol = energy_scale * 1e-9, 1e-10
    result = density_matrix_at_mu(
        h,
        mu=mu,
        kT=kT,
        keys=[(0,)],
        integration=PeriodicGrid(
            density_matrix_tol=1e-3,
            charge_tol=1e-3,
            energy_tol=energy_tol,
            entropy_tol=entropy_tol,
        ),
    )

    def reference(count):
        energies = energy_scale * (0.13 + np.cos(2 * np.pi * np.arange(count) / count))
        f = expit((mu - energies) / kT)
        return np.array([np.mean(energies * f), np.mean(entr(f) + entr(1 - f))])

    exact = reference(32768)
    # Doubling the independent grid establishes substantially tighter accuracy.
    np.testing.assert_allclose(
        exact, reference(65536), rtol=0, atol=energy_scale * 2e-14
    )
    actual_errors = np.abs([result.band_energy, result.entropy] - exact)
    assert np.all(actual_errors <= [energy_tol, entropy_tol]), actual_errors
    assert result.errors.band_energy_integration <= energy_tol
    assert result.errors.entropy_integration <= entropy_tol


@pytest.mark.parametrize("target", ["energy_tol", "entropy_tol"])
def test_thermal_integration_targets_reject_invalid_or_prescribed_requests(target):
    for value in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match=target):
            PeriodicGrid(**{target: value})
    with pytest.raises(ValueError, match="nk cannot be combined"):
        PeriodicGrid(nk=16, **{target: 1e-8})
