"""Root limits apply to brackets, Newton steps and final Brent evaluations."""

import numpy as np
import pytest
from scipy.special import expit

from meanfi.density.filling import solve_mu, solve_mu_in_bracket


def run_root(evaluate, *, bracketed=False, **overrides):
    kwargs = dict(
        filling=0.73,
        mu_guess=0.0,
        filling_tol=1e-12,
        max_charge_evaluations=100,
        use_derivative=True,
    )
    kwargs.update(overrides)
    if bracketed:
        return solve_mu_in_bracket(
            evaluate, lower=-4.0, upper=4.0, mu_xtol=1e-10, **kwargs
        )
    return solve_mu(
        evaluate_charge=evaluate,
        initial_bracket=lambda: (-4.0, 4.0),
        mu_tol=1e-10,
        **kwargs,
    )


@pytest.mark.parametrize("bracketed", [False, True])
@pytest.mark.parametrize("use_derivative", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_root_budget_is_hard_and_failure_has_context(bracketed, use_derivative, budget):
    calls = []

    def evaluate(mu):
        calls.append(mu)
        charge = expit(mu)
        return charge, 0.0, charge * (1 - charge)

    with pytest.raises(
        RuntimeError,
        match="Chemical-potential solve failed: maximum charge-evaluation budget",
    ):
        run_root(
            evaluate,
            bracketed=bracketed,
            max_charge_evaluations=budget,
            use_derivative=use_derivative,
        )
    assert len(calls) == budget


@pytest.mark.parametrize("budget", [0, -1, 2.5, True, np.nan])
def test_root_rejects_invalid_evaluation_budget(budget):
    with pytest.raises(ValueError, match="positive integer"):
        run_root(lambda mu: (expit(mu), 0.0, None), max_charge_evaluations=budget)


@pytest.mark.parametrize("mu_tol", [0.0, -1.0, np.nan, np.inf])
def test_root_rejects_invalid_mu_tolerance(mu_tol):
    with pytest.raises(ValueError, match="mu_tol must be a positive finite"):
        solve_mu(
            evaluate_charge=lambda mu: (expit(mu), 0.0, None),
            initial_bracket=lambda: (-4.0, 4.0),
            filling=0.3,
            mu_guess=0.0,
            filling_tol=1e-6,
            mu_tol=mu_tol,
            max_charge_evaluations=10,
        )
