"""Root limits apply to brackets, Newton steps and final Brent evaluations."""

import numpy as np
import pytest
from scipy.special import expit

from meanfi.density.filling import solve_mu


def run_root(evaluate, **overrides):
    kwargs = dict(
        filling=0.73,
        mu_guess=0.0,
        filling_tol=1e-12,
        max_charge_evaluations=100,
        use_derivative=True,
    )
    kwargs.update(overrides)
    return solve_mu(
        evaluate_charge=evaluate,
        initial_bracket=lambda: (-4.0, 4.0),
        mu_tol=1e-10,
        **kwargs,
    )


@pytest.mark.parametrize("use_derivative", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_root_budget_is_hard_and_failure_has_context(use_derivative, budget):
    calls = []

    def evaluate(mu):
        calls.append(mu)
        charge = expit(mu)
        return charge, charge * (1 - charge)

    with pytest.raises(
        RuntimeError,
        match="Chemical-potential solve failed: maximum charge-evaluation budget",
    ):
        run_root(
            evaluate,
            max_charge_evaluations=budget,
            use_derivative=use_derivative,
        )
    assert len(calls) == budget


@pytest.mark.parametrize("budget", [0, -1, 2.5, True, np.nan])
def test_root_rejects_invalid_evaluation_budget(budget):
    with pytest.raises(ValueError, match="positive integer"):
        run_root(lambda mu: (expit(mu), None), max_charge_evaluations=budget)


@pytest.mark.parametrize("mu_tol", [0.0, -1.0, np.nan, np.inf])
def test_root_rejects_invalid_mu_tolerance(mu_tol):
    with pytest.raises(ValueError, match="mu_tol must be a positive finite"):
        solve_mu(
            evaluate_charge=lambda mu: (expit(mu), None),
            initial_bracket=lambda: (-4.0, 4.0),
            filling=0.3,
            mu_guess=0.0,
            filling_tol=1e-6,
            mu_tol=mu_tol,
            max_charge_evaluations=10,
        )


@pytest.mark.parametrize("use_derivative", [False, True])
def test_accepted_guess_needs_one_charge_evaluation_and_no_bracket(use_derivative):
    calls = []

    def charge(mu):
        calls.append(mu)
        return 0.7, 0.2

    def bracket():
        raise AssertionError("accepted guess must skip bracket construction")

    result = solve_mu(
        evaluate_charge=charge,
        initial_bracket=bracket,
        filling=0.7,
        mu_guess=0.125,
        filling_tol=1e-6,
        mu_tol=1e-10,
        max_charge_evaluations=1,
        use_derivative=use_derivative,
    )
    assert calls == [0.125]
    assert result.mu == 0.125
    assert result.charge_evaluations == 1


def test_root_acceptance_uses_only_requested_filling_residual():
    def bracket():
        pytest.fail("residual already meets the requested target")

    result = solve_mu(
        evaluate_charge=lambda mu: (0.70005, None),
        initial_bracket=bracket,
        filling=0.7,
        mu_guess=0.0,
        filling_tol=1e-4,
        mu_tol=1e-10,
        max_charge_evaluations=1,
    )
    assert result.residual == pytest.approx(5e-5)


@pytest.mark.parametrize("use_derivative", [False, True])
@pytest.mark.parametrize("filling", [0.0, 1.0])
def test_bracket_expansion_accepts_tolerance_without_sign_change(
    use_derivative, filling
):
    calls = []

    def charge(mu):
        calls.append(mu)
        occupation = expit(mu)
        return occupation, occupation * (1 - occupation)

    result = run_root(
        charge,
        filling=filling,
        filling_tol=1e-4,
        max_charge_evaluations=4,
        use_derivative=use_derivative,
    )
    expected_mu = -12.0 if filling == 0.0 else 12.0
    assert calls == [0.0, -4.0, 4.0, expected_mu]
    assert result.mu == expected_mu
    assert result.charge_evaluations == len(calls)
    assert 0.0 < abs(result.residual) <= 1e-4


@pytest.mark.parametrize("filling,budget,expected_mu", [(0.0, 2, -3.0), (2.0, 3, 3.0)])
def test_finite_filling_accepts_bracket_endpoint_within_evaluation_budget(
    filling,
    budget,
    expected_mu,
):
    from meanfi import density_matrix

    energies = np.array([-1.0, 1.0])
    result = density_matrix(
        {(): np.diag(energies)},
        filling,
        kT=0.2,
        keys=[()],
        max_charge_evaluations=budget,
    )
    expected = np.diag(expit((expected_mu - energies) / 0.2))
    error = np.max(np.abs(result.to_tb()[()] - expected))
    assert error < 1e-14, f"Finite density error: {error}"
    assert result.mu == expected_mu
    assert result.errors.filling_residual <= 1e-4
    assert result.statistics.charge_evaluations == budget
