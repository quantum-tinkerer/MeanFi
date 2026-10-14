"""An unattainable frozen charge may propose mu, but cannot pass as converged."""

import numpy as np
import pytest

from meanfi import (
    BlochHamiltonian,
    FermiSimplex,
    default_solver_tolerances,
    density_matrix,
)
from meanfi.density.density import evaluate_density
from meanfi.density.filling import solve_mu
from meanfi.density.integrate.simplex.mesh import SimplexEvaluator
from meanfi.density.problem import build_density_problem
from meanfi.errors import ConvergenceError


pytestmark = pytest.mark.numerics


def step_root(**overrides):
    options = dict(
        evaluate_charge=lambda mu: (float(mu >= 0.5), 0.0),
        initial_bracket=lambda: (-1.0, 1.0),
        filling=0.1,
        mu_guess=-0.5,
        filling_tol=1e-6,
        mu_tol=1e-10,
        max_charge_evaluations=None,
    )
    options.update(overrides)
    return solve_mu(**options)


@pytest.mark.parametrize("use_derivative", [False, True])
@pytest.mark.parametrize("filling", [0.1, 0.5, 0.9])
def test_provisional_root_is_closest_to_target_and_next_to_jump(
    filling, use_derivative
):
    result = step_root(
        allow_inexact=True, filling=filling, use_derivative=use_derivative
    )
    assert not result.converged
    assert abs(result.residual) == pytest.approx(min(filling, 1.0 - filling))
    assert result.charge == float(result.mu >= 0.5)
    assert abs(result.mu - 0.5) < 1e-14


def test_strict_root_still_rejects_an_unattainable_filling():
    with pytest.raises(RuntimeError, match="satisfying the filling tolerance"):
        step_root()


def test_provisional_search_still_marks_a_successful_root_converged():
    result = step_root(allow_inexact=True, filling=0.0)
    assert result.converged
    assert result.residual == 0.0


def test_provisional_search_does_not_hide_a_work_limit():
    with pytest.raises(RuntimeError, match="maximum charge-evaluation budget"):
        step_root(allow_inexact=True, max_charge_evaluations=2)


def test_provisional_search_does_not_hide_a_nonfinite_charge():
    with pytest.raises(RuntimeError, match="non-finite charge"):
        step_root(allow_inexact=True, evaluate_charge=lambda mu: (np.nan, None))


def test_provisional_search_does_not_hide_a_failed_bracket():
    with pytest.raises(ConvergenceError, match="Could not bracket"):
        step_root(allow_inexact=True, evaluate_charge=lambda mu: (0.0, None))


def pocket_problem(*, tol=1e-3, shift=0.0, prescribed=False):
    # The four square corners all have energy shift+1/2, making the frozen
    # interpolant flat. The real occupied disk has exact charge pi*(mu-shift).
    def hamiltonian(x, y):
        radius_squared = (x / (2 * np.pi) - 0.5) ** 2 + (y / (2 * np.pi) - 0.5) ** 2
        return np.array([[shift + radius_squared]])

    return build_density_problem(
        BlochHamiltonian(hamiltonian),
        kT=0.0,
        keys=[(0, 0)],
        integration=FermiSimplex(nk=4)
        if prescribed
        else FermiSimplex(initial_grid_level=0),
        tolerances=default_solver_tolerances(tol),
    )


@pytest.mark.parametrize("filling,tol", [(0.01, 1e-3), (5e-5, 1e-5)])
@pytest.mark.parametrize("shift", [0.0, 2.3])
def test_adaptive_filling_recovers_from_a_flat_starting_band(filling, tol, shift):
    problem = pocket_problem(tol=tol, shift=shift)
    result = evaluate_density(problem, filling=filling)
    exact_charge = np.pi * (result.mu - shift)
    assert 0.0 < result.mu - shift < 0.25
    assert result.errors.charge_integration <= problem.tolerances.charge_integration
    assert result.errors.filling_residual <= problem.tolerances.filling_residual
    assert abs(exact_charge - result.filling) <= result.errors.charge_integration
    assert result.errors.density_matrix_integration <= (
        problem.tolerances.density_matrix_integration
    )
    assert result.statistics.refinements > 0


def test_prescribed_mesh_cannot_accept_a_provisional_filling():
    with pytest.raises(ConvergenceError, match="satisfying the filling tolerance"):
        evaluate_density(pocket_problem(prescribed=True), filling=0.01)


def test_root_budget_is_preserved_across_adaptive_retries(monkeypatch):
    calls = {False: 0, True: 0}
    original_charge = SimplexEvaluator.charge

    def charge(self, mu, *, adaptive):
        calls[adaptive] += 1
        return original_charge(self, mu, adaptive=adaptive)

    monkeypatch.setattr(SimplexEvaluator, "charge", charge)
    with pytest.raises(ConvergenceError, match="maximum charge-evaluation budget"):
        evaluate_density(pocket_problem(), filling=0.01, max_charge_evaluations=100)
    assert calls[False] == 100
    assert calls[True] > 0


def test_genuinely_flat_band_fails_without_repeating_an_unchanged_mesh(monkeypatch):
    original_charge = SimplexEvaluator.charge
    adaptive_calls = 0

    def charge(self, mu, *, adaptive):
        nonlocal adaptive_calls
        if adaptive:
            adaptive_calls += 1
            assert adaptive_calls <= 2, "an unattainable flat-band filling is looping"
        return original_charge(self, mu, adaptive=adaptive)

    monkeypatch.setattr(SimplexEvaluator, "charge", charge)
    with pytest.raises(ConvergenceError, match="stalled on an unchanged mesh"):
        density_matrix({(0,): np.array([[0.5]])}, filling=0.1, keys=[(0,)])
