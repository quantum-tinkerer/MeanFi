from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from meanfi._info import DensityIntegrationInfo
from meanfi._validation import tb_dimension, tb_orbital_count
from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import tb_to_kfunc

if TYPE_CHECKING:
    from stateful_quadrature import StatefulIntegrator
else:
    StatefulIntegrator = Any


def fermi_dirac(energies: np.ndarray, kT: float, fermi: float) -> np.ndarray:
    """Evaluate the Fermi-Dirac distribution."""

    if kT < 0:
        raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")
    if kT == 0:
        energies = np.asarray(energies, dtype=float)
        occupation = np.where(energies < fermi, 1.0, 0.0)
        occupation = np.where(energies == fermi, 0.5, occupation)
        return occupation.astype(float, copy=False)

    occupation = np.empty_like(energies, dtype=float)
    exponent = (energies - fermi) / kT
    sign_mask = energies >= fermi

    pos_exp = np.exp(-exponent[sign_mask])
    neg_exp = np.exp(exponent[~sign_mask])

    occupation[sign_mask] = pos_exp / (pos_exp + 1.0)
    occupation[~sign_mask] = 1.0 / (neg_exp + 1.0)
    return occupation


def quadrature_prefactor(ndim: int) -> float:
    """Return the Brillouin-zone integration prefactor."""

    return 1.0 if ndim == 0 else 1.0 / (2.0 * np.pi) ** ndim


def integration_stats(result) -> DensityIntegrationInfo:
    """Convert integrator metadata to the public info dataclass."""

    cached_nodes = getattr(result, "n_cached_nodes", getattr(result, "n_leaf_nodes", 0))
    return DensityIntegrationInfo(
        n_kernel_evals=int(result.n_kernel_evals),
        n_evaluator_evals=int(result.n_evaluator_evals),
        n_cached_nodes=int(cached_nodes),
        n_leaves=int(getattr(result, "n_leaves", 0)),
        n_leaf_nodes=int(getattr(result, "n_leaf_nodes", cached_nodes)),
        subdivisions=int(getattr(result, "subdivisions", 0)),
    )


def spectral_payload(hamiltonian: _tb_type):
    """Build the cached eigensystem payload consumed by stateful quadrature."""

    hkfunc = tb_to_kfunc(hamiltonian)
    ndof = tb_orbital_count(hamiltonian)

    def kernel(points: np.ndarray) -> np.ndarray:
        h_k = hkfunc(points)
        eigenvalues, eigenvectors = np.linalg.eigh(h_k)
        return np.concatenate(
            [eigenvalues, eigenvectors.reshape(points.shape[0], ndof * ndof)],
            axis=-1,
        )

    return kernel


def charge_evaluator(ndim: int, ndof: int, kT: float):
    """Return an evaluator for charge and charge-derivative integrands."""

    prefactor = quadrature_prefactor(ndim)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        del points
        eigenvalues = payload[:, :ndof].real
        occupation = fermi_dirac(eigenvalues, kT, mu)
        dcharge_dmu = occupation * (1.0 - occupation) / kT
        charge = np.sum(occupation, axis=-1, keepdims=True)
        derivative = np.sum(dcharge_dmu, axis=-1, keepdims=True)
        return prefactor * np.concatenate([charge, derivative], axis=-1)

    return evaluator


def density_evaluator(ndim: int, ndof: int, keys: list[tuple[int, ...]], kT: float):
    """Return an evaluator for density-matrix Fourier components."""

    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(keys, dtype=float)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        eigenvalues = payload[:, :ndof].real
        eigenvectors = payload[:, ndof:].reshape(points.shape[0], ndof, ndof)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density_k = (
            eigenvectors
            * occupation[:, np.newaxis, :]
            @ eigenvectors.conj().transpose(0, 2, 1)
        )
        phase = np.exp(1j * np.dot(points, keys_array.T))
        density_terms = (
            density_k[..., np.newaxis] * phase[:, np.newaxis, np.newaxis, :]
        ).reshape(points.shape[0], -1)
        return prefactor * density_terms

    return evaluator


def split_charge_result(
    estimate: np.ndarray, error: np.ndarray
) -> tuple[float, float, float]:
    """Return charge, charge error, and dcharge/dmu from a quadrature result."""

    estimate = np.asarray(estimate)
    error = np.asarray(error)
    return (
        float(np.real(estimate[0])),
        float(np.abs(error[0])),
        float(np.real(estimate[1])),
    )


def split_density_result(
    estimate: np.ndarray,
    error: np.ndarray,
    ndof: int,
    keys: list[tuple[int, ...]],
) -> tuple[_tb_type, _tb_type]:
    """Convert vectorized density estimates to tight-binding dictionaries."""

    estimate = np.asarray(estimate).reshape(ndof, ndof, len(keys))
    error = np.asarray(error).reshape(ndof, ndof, len(keys))

    rho = {}
    rho_error = {}
    for index, key in enumerate(keys):
        rho[key] = estimate[..., index]
        rho_error[key] = error[..., index]
    return rho, rho_error


def mu_bracket(hamiltonian: _tb_type, kT: float) -> tuple[float, float]:
    """Return a conservative chemical-potential bracket."""

    bound = sum(np.linalg.norm(matrix, ord=2) for matrix in hamiltonian.values())
    padding = max(1.0, 10.0 * kT)
    return -float(bound + padding), float(bound + padding)


def charge_integral_tolerance(charge_tol: float) -> tuple[float, float]:
    """Translate charge tolerance to the charge-integral tolerance pair."""

    return float(charge_tol) / 4.0, 0.0


def integration_bounds(ndim: int) -> tuple[list[float], list[float]]:
    """Return the Brillouin-zone bounds for stateful quadrature."""

    return ([-np.pi] * ndim, [np.pi] * ndim)


def build_integrator(
    hamiltonian: _tb_type,
    *,
    evaluator,
    rule: str,
    batch_size: int | None,
):
    """Construct a stateful quadrature integrator for a tight-binding model."""

    try:
        from stateful_quadrature import StatefulIntegrator
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "Finite-temperature integration requires the optional stateful_quadrature dependency"
        ) from exc

    a, b = integration_bounds(tb_dimension(hamiltonian))
    return StatefulIntegrator(
        a=a,
        b=b,
        kernel=spectral_payload(hamiltonian),
        evaluator=evaluator,
        rule=rule,
        batch_size=batch_size,
    )


def run_integrator(
    integrator: StatefulIntegrator,
    parameter: float,
    *,
    atol: float,
    rtol: float,
    max_subdivisions: int | None,
    error_message: str,
):
    """Run the adaptive integrator and normalize non-convergence errors."""

    result = integrator.integrate(
        parameter,
        atol=atol,
        rtol=rtol,
        max_subdivisions=max_subdivisions,
    )
    if result.status != "converged":
        raise ValueError(error_message)
    return result


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
    max_mu_iterations: int,
) -> tuple[float, float, float, float, int]:
    """Solve for the chemical potential using safeguarded Newton steps."""

    mu = float(np.clip(mu_guess, lower, upper))
    if not lower < mu < upper:
        mu = 0.5 * (lower + upper)

    last_charge = float("nan")
    last_charge_error = float("nan")
    last_derivative = float("nan")
    for iteration in range(1, max_mu_iterations + 1):
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

        candidate = mu - residual / last_derivative if last_derivative > 0 else np.nan
        if not np.isfinite(candidate) or candidate <= lower or candidate >= upper:
            candidate = 0.5 * (lower + upper)
        mu = float(candidate)

    raise ValueError(
        "Chemical-potential solver did not converge within max_mu_iterations"
    )
