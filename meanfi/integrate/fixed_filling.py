from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from meanfi.core.filling import expand_mu_bracket, solve_mu


@dataclass(frozen=True)
class FixedFillingSolve:
    mu: float
    charge: float
    charge_error: float
    derivative: float
    root_iterations: int


def solve_fixed_filling_root(
    *,
    evaluate_charge: Callable[[float], tuple[float, float, float]],
    mu_bracket: Callable[[], tuple[float, float]],
    filling: float,
    mu_guess: float,
    filling_tol: float,
    mu_tol: float,
    max_mu_iterations: int | None,
) -> FixedFillingSolve:
    lower, upper = mu_bracket()
    lower, upper = expand_mu_bracket(
        evaluate_charge,
        filling=filling,
        lower=lower,
        upper=upper,
    )
    mu, charge, charge_error, derivative, root_iterations = solve_mu(
        evaluate_charge,
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=filling_tol,
        mu_xtol=mu_tol,
        max_mu_iterations=max_mu_iterations,
    )
    return FixedFillingSolve(
        mu=float(mu),
        charge=float(charge),
        charge_error=float(charge_error),
        derivative=float(derivative),
        root_iterations=int(root_iterations),
    )
