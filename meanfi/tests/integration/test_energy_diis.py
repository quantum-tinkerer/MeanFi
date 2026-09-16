from __future__ import annotations

import numpy as np
import pytest

from meanfi import (
    UniformGrid,
    FermiSimplex,
    EnergyDIIS,
    ErrorTolerances,
    Model,
    solver,
)
from meanfi.scf.ediis import EDIISPoint, ediis_coefficients


pytestmark = pytest.mark.integration


def test_ediis_minimizes_exact_quadratic_energy_on_convex_hull():
    history = [
        EDIISPoint(np.array([0.0]), 1.0),
        EDIISPoint(np.array([2.0]), 1.0),
    ]

    coefficients = ediis_coefficients(
        history,
        interaction_curvature=lambda difference: float(difference[0] ** 2),
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
        integration=FermiSimplex(density_matrix_tol=1e-6),
        scf_tol=1e-7,
    )

    assert result.converged is True
    assert result.history
    assert not hasattr(result, "accuracy")
    assert result.internal_energy == pytest.approx(-0.5)


def test_explicit_energy_diis_uses_requested_tolerances_from_first_iteration():
    result = solver(
        _zero_dimensional_model(),
        {(): np.zeros((2, 2), dtype=complex)},
        integration=FermiSimplex(density_matrix_tol=1e-6),
        scf=EnergyDIIS(),
        scf_tol=1e-7,
    )

    assert len(result.history) == 1
    assert result.internal_energy == pytest.approx(-0.5)


def test_energy_diis_evaluates_the_tolerance_policy_once_for_the_solve():
    calls = []

    def custom_tolerances(tol):
        calls.append(tol)
        return ErrorTolerances(
            scf_residual=tol,
            density_matrix_integration=tol / 20,
            filling_residual=tol / 5,
            charge_integration=tol / 50,
            matrix_function_tol=tol / 5,
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
        matrix_function_tol=2e-7,
    )
    assert calls == [1e-6]
    assert result.errors.scf_residual <= requested.scf_residual
    assert result.errors.density_matrix_integration <= (
        requested.density_matrix_integration
    )
    assert result.errors.filling_residual <= requested.filling_residual


def test_finite_temperature_default_reports_free_energy():
    result = solver(
        _zero_dimensional_model(kT=0.2),
        {(): np.zeros((2, 2), dtype=complex)},
        integration=UniformGrid(density_matrix_tol=1e-8),
        scf_tol=1e-7,
    )

    assert result.history
    assert np.isfinite(result.internal_energy)
    assert result.free_energy == pytest.approx(
        result.internal_energy - 0.2 * result.entropy
    )


def test_energy_diis_supports_periodic_integration():
    result = solver(
        _zero_dimensional_model(kT=0.2),
        {(): np.zeros((2, 2), dtype=complex)},
        integration=UniformGrid(),
        scf=EnergyDIIS(),
    )
    assert result.converged
    assert np.isfinite(result.free_energy)


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
        integration=FermiSimplex(
            density_matrix_tol=2e-3,
            max_refinements=500,
        ),
        scf=EnergyDIIS(),
        scf_tol=3e-3,
    )

    assert np.isfinite(result.internal_energy)
    assert all(np.isfinite(item.internal_energy) for item in result.history)
    assert result.errors.scf_residual <= 3e-3


def test_default_zero_temperature_solver_uses_ediis(monkeypatch):
    import meanfi.scf.engine as engine

    def unexpected_fixed_point(*args, **kwargs):
        raise AssertionError("normal zero-temperature default must use EDIIS")

    monkeypatch.setattr(engine, "iterate_anderson", unexpected_fixed_point)
    result = solver(_zero_dimensional_model(), {(): np.zeros((2, 2))})
    assert result.converged


@pytest.mark.parametrize("kind", ["normal", "reference", "bdg"])
def test_scf_interaction_functional_matches_exact_two_orbital_energy(kind):
    """For two orbitals, Wick's theorem gives the full interaction polynomial."""
    from meanfi.density.problem import build_density_problem
    from meanfi.errors import default_solver_tolerances
    from meanfi.scf.problem import SCFProblem
    from meanfi.space.state import ActiveDensityState
    from meanfi.tests.fixtures.models import density_result_from_tb

    reference = None
    if kind == "reference":
        reference = density_result_from_tb(
            {(): np.array([[0.3, 0.02j], [-0.02j, 0.6]])}
        )
    interaction = 1.7
    model = Model(
        {(): np.diag([-0.3, 0.2])},
        {(): np.array([[0.0, interaction], [interaction, 0.0]])},
        filling=0.9,
        kT=0.2,
        superconducting=kind == "bdg",
        reference=reference,
    )
    problem = SCFProblem(
        model,
        build_density_problem(
            model.hamiltonian_from_meanfield(),
            kT=model.kT,
            keys=[()],
            integration=UniformGrid(),
            tolerances=default_solver_tolerances(1e-10),
            density_coordinates=model.required_coordinates,
            electron_ndof=2 if model.superconducting else None,
        ),
    )
    rng = np.random.default_rng(421)
    params = rng.uniform(-0.5, 0.5, model._space.num_params)
    direction = rng.normal(size=params.size)

    def exact_energy(values):
        density = model._active_density_from_state(
            ActiveDensityState(model._space, values)
        )[()]
        electron = density[:2, :2]
        if reference is not None:
            electron = electron - reference.to_tb()[()]
        pairing = abs(density[0, 3]) ** 2 if model.superconducting else 0.0
        return float(
            interaction
            * (
                electron[0, 0].real * electron[1, 1].real
                - abs(electron[0, 1]) ** 2
                - pairing
            )
            / 2
        )

    step = 1e-5
    exact_gradient = (
        exact_energy(params + step * direction)
        - exact_energy(params - step * direction)
    ) / (2 * step)
    from meanfi.meanfield import (
        interaction_energy,
        correction_expectation,
        interaction_correction,
    )

    state = ActiveDensityState(model._space, params)
    difference = model._active_density_from_state(model._reference_difference(state))
    assert interaction_energy(
        difference, model.h_int, electron_ndof=model._electron_ndof
    ) == pytest.approx(exact_energy(params), abs=1e-13)
    # Central differences are exact for this quadratic, up to floating-point error.
    correction = interaction_correction(
        difference, model.h_int, electron_ndof=model._electron_ndof
    )
    assert correction_expectation(
        model._space.density_from_params(direction),
        correction,
        electron_ndof=model._electron_ndof,
    ) == pytest.approx(exact_gradient, abs=2e-10)
    assert problem.interaction_curvature(direction) == pytest.approx(
        (
            exact_energy(params + direction)
            + exact_energy(params - direction)
            - 2 * exact_energy(params)
        )
        / 2,
        abs=1e-13,
    )


def test_ediis_quadratic_history_matches_exact_minimum_with_reference_offset():
    rng = np.random.default_rng(427)
    matrix = rng.normal(size=(3, 3))
    hessian = matrix.T @ matrix + np.eye(3)
    target = np.array([0.15, 0.25, 0.1])
    vertices = np.vstack([np.zeros(3), np.eye(3)])

    def exact_energy(params):
        difference = params - target
        return 17.0 + 0.5 * difference @ hessian @ difference

    history = [EDIISPoint(params, exact_energy(params)) for params in vertices]
    coefficients = ediis_coefficients(
        history,
        interaction_curvature=lambda difference: float(
            0.5 * difference @ hessian @ difference
        ),
    )
    # The target is inside this simplex; its energy is the exact global minimum.
    mixed = coefficients @ vertices
    np.testing.assert_allclose(mixed, target, atol=1e-6, rtol=0)
    assert abs(exact_energy(mixed) - 17.0) < 1e-12


def test_ediis_trajectory_is_independent_of_reported_entropy(monkeypatch):
    from dataclasses import replace
    import meanfi.scf.problem as scf_problem

    model = _zero_dimensional_model(kT=0.2)
    guess = {(): np.zeros((2, 2))}
    baseline = solver(model, guess, tol=1e-9)
    evaluate_density = scf_problem.evaluate_density

    def altered_entropy(*args, **kwargs):
        density = evaluate_density(*args, **kwargs)
        return replace(density, entropy=1000 * float(np.linalg.norm(density.values)))

    monkeypatch.setattr(scf_problem, "evaluate_density", altered_entropy)
    altered = solver(model, guess, tol=1e-9)
    assert len(baseline.history) > 1
    assert altered.history == baseline.history
    np.testing.assert_array_equal(altered.density.values, baseline.density.values)
    assert altered.internal_energy == baseline.internal_energy
    assert altered.free_energy != baseline.free_energy
    assert altered.free_energy == pytest.approx(
        altered.internal_energy - model.kT * altered.entropy
    )
