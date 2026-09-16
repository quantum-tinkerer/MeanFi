from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from meanfi.density.problem import build_density_problem

from meanfi import (
    FermiSimplex,
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
        density_matrix_integration=2e-4,
        filling_residual=1e-4,
        charge_integration=2e-4,
        matrix_function_tol=2e-4,
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


@pytest.mark.parametrize("charge_tolerance", [2e-7, 2e-4])
def test_explicit_integration_tolerances_are_effective_internal_requests(
    charge_tolerance,
):
    problem = build_density_problem(
        _two_level_hamiltonian(),
        kT=0.2,
        keys=[()],
        integration=UniformGrid(
            density_matrix_tol=5e-7,
            charge_tol=charge_tolerance,
        ),
        tolerances=default_solver_tolerances(1e-4),
    )

    assert problem.tolerances == ErrorTolerances(
        scf_residual=1e-4,
        density_matrix_integration=5e-7,
        filling_residual=1e-5,
        charge_integration=charge_tolerance,
        matrix_function_tol=2e-5,
    )


def test_unavailable_periodic_grid_estimators_are_none():
    result = density_matrix(
        {(0,): np.diag([-0.5, 0.5])},
        filling=1.0,
        kT=0.2,
        keys=[(0,)],
        integration=UniformGrid(nk=8),
        tol=1e-3,
    )

    assert result.errors == ErrorValues(
        filling_residual=result.errors.filling_residual,
    )
    assert result.errors.scf_residual is None
    assert result.errors.density_matrix_integration is None
    assert result.errors.charge_integration is None


def test_energy_units_do_not_control_density_refinement():
    from scipy.special import entr, expit
    from meanfi import density_matrix_at_mu

    h = {(0,): np.array([[0.13]]), (1,): np.array([[0.5]]), (-1,): np.array([[0.5]])}
    tolerance = 1e-3
    results = []
    for scale in (1.0, 1e6):
        results.append(
            density_matrix_at_mu(
                {key: scale * block for key, block in h.items()},
                mu=scale * 0.27,
                kT=scale * 0.037,
                keys=[(0,)],
                integration=UniformGrid(
                    density_matrix_tol=tolerance, charge_tol=tolerance
                ),
            )
        )
    base, scaled = results
    assert base.statistics.n_kpoints == scaled.statistics.n_kpoints
    np.testing.assert_allclose(
        base.entries.values, scaled.entries.values, atol=1e-12, rtol=0
    )

    def reference(count):
        energies = 0.13 + np.cos(2 * np.pi * np.arange(count) / count)
        occupations = expit((0.27 - energies) / 0.037)
        return np.array(
            [
                np.mean(occupations),
                np.mean(energies * occupations),
                np.mean(entr(occupations) + entr(1 - occupations)),
            ]
        )

    exact = reference(32768)
    np.testing.assert_allclose(exact, reference(65536), atol=2e-14, rtol=0)
    actual = [base.entries.values[0].real, base.band_energy, base.entropy]
    errors = np.abs(actual - exact)
    assert errors[0] <= tolerance, errors
    # These are checks against an independently converged reference, not solver
    # stopping criteria. Their scale changes with the Hamiltonian's energy units.
    assert np.all(errors[1:] <= 1e-3), errors
    assert scaled.errors.band_energy_integration > tolerance
    np.testing.assert_allclose(
        scaled.band_energy / 1e6, base.band_energy, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(scaled.entropy, base.entropy, atol=1e-12, rtol=0)


@pytest.mark.parametrize("method,kT", [(FermiSimplex, 0.0), (UniformGrid, 0.2)])
def test_explicit_density_target_supplies_omitted_charge_target(method, kT):
    problem = build_density_problem(
        _two_level_hamiltonian(),
        kT=kT,
        keys=[()],
        integration=method(density_matrix_tol=5e-7),
        tolerances=default_solver_tolerances(1e-3),
    )
    assert problem.tolerances.density_matrix_integration == 5e-7
    assert problem.tolerances.charge_integration == 5e-7


def test_custom_charge_policy_is_retained_without_mesh_overrides():
    tolerances = replace(default_solver_tolerances(1e-3), charge_integration=1e-2)
    problem = build_density_problem(
        _two_level_hamiltonian(),
        kT=0.2,
        keys=[()],
        integration=UniformGrid(),
        tolerances=tolerances,
    )
    assert problem.tolerances == tolerances


@pytest.mark.parametrize("target", [1e-3, 1e-9])
@pytest.mark.usefixtures("require_mumps")
def test_matrix_function_budget_and_report_match_dense_reference(target):
    from scipy import sparse
    from scipy.special import expit
    from meanfi import density_matrix_at_mu, RationalFOE

    n = 24
    matrix = sparse.diags(
        [-np.ones(n - 1), np.linspace(-0.3, 0.4, n), -np.ones(n - 1)],
        [-1, 0, 1],
        format="csr",
        dtype=complex,
    )
    requested = replace(
        default_solver_tolerances(1e-3),
        matrix_function_tol=target,
        density_matrix_integration=1e-12,
        charge_integration=1e-12,
        filling_residual=1e-12,
    )
    result = density_matrix_at_mu(
        {(): matrix},
        mu=0.13,
        kT=0.2,
        keys=[()],
        integration=UniformGrid(matrix_function=RationalFOE()),
        tolerance_policy=lambda _: requested,
    )
    energies, vectors = np.linalg.eigh(matrix.toarray())
    exact = (vectors * expit((0.13 - energies) / 0.2)) @ vectors.conj().T
    error = np.max(abs(result.to_tb()[()] - exact))
    estimated = result.errors.matrix_function_error
    assert 0 <= estimated <= target
    assert error <= 1.01 * estimated + 1e-12, (error, estimated)
    assert result.errors.density_matrix_integration == 0.0
    if target == 1e-3:
        # Neither integration nor an unused filling-root target tightens this fit.
        assert estimated > 1e-8


@pytest.mark.parametrize("method,kT", [(FermiSimplex(), 0), (UniformGrid(), 0.2)])
def test_direct_density_does_not_invent_a_matrix_function_estimate(method, kT):
    from meanfi import density_matrix_at_mu, IntegrationInfo

    result = density_matrix_at_mu(
        _two_level_hamiltonian(),
        0,
        kT=kT,
        keys=[()],
        integration=method,
    )
    assert isinstance(result.statistics, IntegrationInfo)
    assert result.errors.matrix_function_error is None
