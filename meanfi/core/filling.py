from __future__ import annotations

import numpy as np

from meanfi.core.matrix import to_dense
from meanfi.tb.tb import _tb_type


def mu_bracket(hamiltonian: _tb_type, kT: float) -> tuple[float, float]:
    """Return a conservative chemical-potential bracket."""

    bound = sum(np.linalg.norm(to_dense(matrix), ord=2) for matrix in hamiltonian.values())
    padding = max(1.0, 10.0 * kT)
    return -float(bound + padding), float(bound + padding)


def charge_integral_tolerance(charge_tol: float) -> tuple[float, float]:
    """Translate charge tolerance to the charge-integral tolerance pair."""

    return float(charge_tol) / 4.0, 0.0


def expand_mu_bracket(
    evaluate_charge,
    *,
    filling: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Expand a charge bracket until it encloses the requested filling."""

    lower_charge, _, _ = evaluate_charge(lower)
    upper_charge, _, _ = evaluate_charge(upper)
    while lower_charge > filling or upper_charge < filling:
        lower *= 2.0
        upper *= 2.0
        lower_charge, _, _ = evaluate_charge(lower)
        upper_charge, _, _ = evaluate_charge(upper)
    return lower, upper


def solve_mu(
    evaluate_charge,
    *,
    filling: float,
    mu_guess: float,
    lower: float,
    upper: float,
    charge_tol: float,
    mu_xtol: float,
    max_mu_iterations: int | None,
) -> tuple[float, float, float, float, int]:
    """Solve for the chemical potential using safeguarded Newton steps."""

    mu = float(np.clip(mu_guess, lower, upper))
    if not lower < mu < upper:
        mu = 0.5 * (lower + upper)

    last_charge = float("nan")
    last_charge_error = float("nan")
    last_derivative = float("nan")
    iteration = 0
    while True:
        iteration += 1
        last_charge, last_charge_error, last_derivative = evaluate_charge(mu)
        residual = last_charge - filling
        if abs(residual) <= charge_tol and last_charge_error <= charge_tol / 2.0:
            return mu, last_charge, last_charge_error, last_derivative, iteration

        if residual < 0:
            lower = mu
        else:
            upper = mu

        if upper - lower <= mu_xtol:
            mu = 0.5 * (lower + upper)
            continue

        if last_derivative <= 0 or not np.isfinite(last_derivative):
            next_mu = 0.5 * (lower + upper)
        else:
            next_mu = mu - residual / last_derivative
            if not lower < next_mu < upper:
                next_mu = 0.5 * (lower + upper)
        mu = float(next_mu)

        if max_mu_iterations is not None and iteration >= max_mu_iterations:
            return mu, last_charge, last_charge_error, last_derivative, iteration
