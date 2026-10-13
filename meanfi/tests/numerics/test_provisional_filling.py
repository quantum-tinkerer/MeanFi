"""An unattainable frozen charge may propose mu, but cannot pass as converged."""

import numpy as np
import pytest

from meanfi.density.filling import solve_mu


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


def test_provisional_root_is_closest_to_target_and_next_to_jump():
    result = step_root(allow_inexact=True)
    assert not result.converged
    assert result.charge == 0.0
    assert result.residual == pytest.approx(-0.1)
    assert result.mu < 0.5
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


@pytest.mark.parametrize("filling,tol", [(0.01, 1e-3), (5e-5, 1e-5)])
@pytest.mark.parametrize("shift", [0.0, 2.3])
def test_adaptive_filling_recovers_from_a_flat_starting_band(filling, tol, shift):
    from meanfi import BlochHamiltonian, FermiSimplex, default_solver_tolerances
    from meanfi.density.density import evaluate_density
    from meanfi.density.problem import build_density_problem

    # The four square corners all have energy shift+1/2, making the frozen
    # interpolant flat. The real occupied disk has exact charge pi*(mu-shift).
    hamiltonian = BlochHamiltonian(
        lambda x, y: np.array(
            [[shift + (x / (2 * np.pi) - 0.5) ** 2
              + (y / (2 * np.pi) - 0.5) ** 2]]
        )
    )
    problem = build_density_problem(
        hamiltonian,
        kT=0.0,
        keys=[(0, 0)],
        integration=FermiSimplex(initial_nk=4),
        tolerances=default_solver_tolerances(tol),
    )
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
