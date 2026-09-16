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
