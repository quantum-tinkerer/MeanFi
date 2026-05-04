from __future__ import annotations

import numpy as np

from meanfi.tb.ops import hermitian_spectral_bound, is_sparse_like, matrix_bound
from meanfi.tb.ops import _tb_type


def _tb_k_matrix(hamiltonian: _tb_type, point: np.ndarray):
    accumulator = None
    for key, matrix in hamiltonian.items():
        phase = np.exp(-1j * np.dot(point, np.asarray(key, dtype=float)))
        term = matrix * phase
        accumulator = term if accumulator is None else accumulator + term
    if accumulator is None:
        raise ValueError("Hamiltonian cannot be empty")
    return accumulator


def _spectral_probe_points(ndim: int) -> list[np.ndarray]:
    if ndim == 0:
        return [np.zeros(0, dtype=float)]

    points = [np.zeros(ndim, dtype=float), np.full(ndim, np.pi, dtype=float)]
    for axis in range(ndim):
        point = np.zeros(ndim, dtype=float)
        point[axis] = np.pi
        points.append(point)

    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.array_equal(point, existing) for existing in unique):
            unique.append(point)
    return unique


def _initial_spectral_bound(hamiltonian: _tb_type) -> float:
    fallback = sum(matrix_bound(matrix) for matrix in hamiltonian.values())
    if fallback == 0.0:
        return 0.0

    if not any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
        return fallback

    ndim = len(next(iter(hamiltonian)))
    estimate = 0.0
    for point in _spectral_probe_points(ndim):
        estimate = max(estimate, hermitian_spectral_bound(_tb_k_matrix(hamiltonian, point)))
    if estimate == 0.0:
        return fallback
    return float(min(fallback, estimate + 1e-3 * max(1.0, estimate)))


def mu_bracket(hamiltonian: _tb_type, kT: float) -> tuple[float, float]:
    """Return a conservative chemical-potential bracket."""

    bound = _initial_spectral_bound(hamiltonian)
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


def solve_mu_charge_only(
    evaluate_charge,
    *,
    filling: float,
    mu_guess: float,
    lower: float,
    upper: float,
    charge_tol: float,
    mu_xtol: float,
    max_mu_iterations: int | None,
) -> tuple[float, float, float, None, int]:
    """Solve for the chemical potential using safeguarded bracketing only."""

    lower_charge, _lower_error, _ = evaluate_charge(lower)
    upper_charge, _upper_error, _ = evaluate_charge(upper)
    if lower_charge > filling or upper_charge < filling:
        raise ValueError("Chemical-potential bracket does not enclose the requested filling")

    mu = float(np.clip(mu_guess, lower, upper))
    last_charge = float("nan")
    last_charge_error = float("nan")
    iteration = 0

    if lower < mu < upper:
        iteration += 1
        last_charge, last_charge_error, _ = evaluate_charge(mu)
        residual = last_charge - filling
        if abs(residual) <= charge_tol and last_charge_error <= charge_tol / 2.0:
            return mu, last_charge, last_charge_error, None, iteration
        if residual < 0:
            lower = mu
            lower_charge = last_charge
        else:
            upper = mu
            upper_charge = last_charge

    while True:
        iteration += 1
        denominator = upper_charge - lower_charge
        if denominator > 0 and np.isfinite(denominator):
            mu = lower + (filling - lower_charge) * (upper - lower) / denominator
        else:
            mu = 0.5 * (lower + upper)
        if not lower < mu < upper:
            mu = 0.5 * (lower + upper)
        last_charge, last_charge_error, _ = evaluate_charge(mu)
        residual = last_charge - filling
        if abs(residual) <= charge_tol and last_charge_error <= charge_tol / 2.0:
            return mu, last_charge, last_charge_error, None, iteration

        if residual < 0:
            lower = mu
            lower_charge = last_charge
        else:
            upper = mu
            upper_charge = last_charge

        if upper - lower <= mu_xtol:
            return 0.5 * (lower + upper), last_charge, last_charge_error, None, iteration

        if max_mu_iterations is not None and iteration >= max_mu_iterations:
            return mu, last_charge, last_charge_error, None, iteration


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
