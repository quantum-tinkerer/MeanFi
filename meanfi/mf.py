from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from meanfi.tb.tb import add_tb, _tb_type
from meanfi.tb.transforms import tb_to_kfunc

if TYPE_CHECKING:
    from stateful_quadrature import StatefulIntegrator
else:
    StatefulIntegrator = Any


def fermi_dirac(E: np.ndarray, kT: float, fermi: float) -> np.ndarray:
    """Evaluate the Fermi-Dirac distribution."""
    if kT < 0:
        raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")
    if kT == 0:
        energies = np.asarray(E, dtype=float)
        occupation = np.where(energies < fermi, 1.0, 0.0)
        occupation = np.where(energies == fermi, 0.5, occupation)
        return occupation.astype(float, copy=False)

    fd = np.empty_like(E, dtype=float)
    exponent = (E - fermi) / kT
    sign_mask = E >= fermi

    pos_exp = np.exp(-exponent[sign_mask])
    neg_exp = np.exp(exponent[~sign_mask])

    fd[sign_mask] = pos_exp / (pos_exp + 1.0)
    fd[~sign_mask] = 1.0 / (neg_exp + 1.0)
    return fd


@dataclass(frozen=True)
class DensityIntegrationInfo:
    """Statistics for a single density integration at fixed chemical potential."""

    n_kernel_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int


@dataclass(frozen=True)
class FixedFillingInfo:
    """Statistics for a fixed-filling density calculation."""

    mu: float
    charge: float
    charge_error: float
    dcharge_dmu: float
    root_iterations: int
    charge_integration_calls: int
    density_integration_calls: int
    charge_n_kernel_evals: int
    density_n_kernel_evals: int
    n_kernel_evals: int
    charge_n_evaluator_evals: int
    density_n_evaluator_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int
    charge_integral_atol: float
    density_atol: float
    density_rtol: float


def _quadrature_prefactor(ndim: int) -> float:
    return 1.0 if ndim == 0 else 1.0 / (2.0 * np.pi) ** ndim


def _normalize_keys(h: _tb_type, keys: list) -> list[tuple[int, ...]]:
    keys = [tuple(key) for key in keys]
    ndim = len(next(iter(h)))
    if any(len(key) != ndim for key in keys):
        raise ValueError("All keys must have the same dimension as the Hamiltonian")
    return keys


def _integration_stats(result) -> DensityIntegrationInfo:
    cached_nodes = getattr(result, "n_cached_nodes", getattr(result, "n_leaf_nodes", 0))
    return DensityIntegrationInfo(
        n_kernel_evals=int(result.n_kernel_evals),
        n_evaluator_evals=int(result.n_evaluator_evals),
        n_cached_nodes=int(cached_nodes),
        n_leaves=int(getattr(result, "n_leaves", 0)),
        n_leaf_nodes=int(getattr(result, "n_leaf_nodes", cached_nodes)),
        subdivisions=int(getattr(result, "subdivisions", 0)),
    )


def _spectral_payload(h: _tb_type):
    hkfunc = tb_to_kfunc(h)
    ndof = next(iter(h.values())).shape[0]

    def kernel(points: np.ndarray) -> np.ndarray:
        h_k = hkfunc(points)
        eigenvalues, eigenvectors = np.linalg.eigh(h_k)
        return np.concatenate(
            [eigenvalues, eigenvectors.reshape(points.shape[0], ndof * ndof)], axis=-1
        )

    return kernel


def _charge_evaluator(ndim: int, ndof: int, kT: float):
    prefactor = _quadrature_prefactor(ndim)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        del points
        eigenvalues = payload[:, :ndof].real
        occupation = fermi_dirac(eigenvalues, kT, mu)
        dcharge_dmu = occupation * (1.0 - occupation) / kT
        charge = np.sum(occupation, axis=-1, keepdims=True)
        derivative = np.sum(dcharge_dmu, axis=-1, keepdims=True)
        return prefactor * np.concatenate([charge, derivative], axis=-1)

    return evaluator


def _density_evaluator(ndim: int, ndof: int, keys: list[tuple[int, ...]], kT: float):
    prefactor = _quadrature_prefactor(ndim)
    keys_arr = np.array(keys, dtype=float)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        eigenvalues = payload[:, :ndof].real
        eigenvectors = payload[:, ndof:].reshape(points.shape[0], ndof, ndof)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density_k = (
            eigenvectors
            * occupation[:, np.newaxis, :]
            @ eigenvectors.conj().transpose(0, 2, 1)
        )
        phase = np.exp(1j * np.dot(points, keys_arr.T))
        density_terms = (
            density_k[..., np.newaxis] * phase[:, np.newaxis, np.newaxis, :]
        ).reshape(points.shape[0], -1)
        return prefactor * density_terms

    return evaluator


def _split_charge_result(estimate: np.ndarray, error: np.ndarray) -> tuple[float, float, float]:
    estimate = np.asarray(estimate)
    error = np.asarray(error)
    return (
        float(np.real(estimate[0])),
        float(np.abs(error[0])),
        float(np.real(estimate[1])),
    )


def _split_density_result(
    estimate: np.ndarray, error: np.ndarray, ndof: int, keys: list[tuple[int, ...]]
) -> tuple[_tb_type, _tb_type]:
    estimate = np.asarray(estimate).reshape(ndof, ndof, len(keys))
    error = np.asarray(error).reshape(ndof, ndof, len(keys))
    density = {}
    density_error = {}
    for idx, key in enumerate(keys):
        density[key] = estimate[..., idx]
        density_error[key] = error[..., idx]
    return density, density_error


def _mu_bracket(h: _tb_type, kT: float) -> tuple[float, float]:
    bound = sum(np.linalg.norm(matrix, ord=2) for matrix in h.values())
    padding = max(1.0, 10.0 * kT)
    return -float(bound + padding), float(bound + padding)


def _charge_integral_tolerance(charge_tol: float) -> tuple[float, float]:
    return float(charge_tol) / 4.0, 0.0


def _integration_bounds(ndim: int) -> tuple[list[float], list[float]]:
    return ([-np.pi] * ndim, [np.pi] * ndim)


def _build_integrator(
    h: _tb_type,
    *,
    evaluator,
    rule: str,
    batch_size: int | None,
):
    try:
        from stateful_quadrature import StatefulIntegrator
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "Finite-temperature integration requires the optional stateful_quadrature dependency"
        ) from exc

    ndim = len(next(iter(h)))
    a, b = _integration_bounds(ndim)
    return StatefulIntegrator(
        a=a,
        b=b,
        kernel=_spectral_payload(h),
        evaluator=evaluator,
        rule=rule,
        batch_size=batch_size,
    )


def _run_integrator(
    integrator: StatefulIntegrator,
    parameter: float,
    *,
    atol: float,
    rtol: float,
    max_subdivisions: int | None,
    error_message: str,
):
    result = integrator.integrate(
        parameter,
        atol=atol,
        rtol=rtol,
        max_subdivisions=max_subdivisions,
    )
    if result.status != "converged":
        raise ValueError(error_message)
    return result


def _expand_mu_bracket(evaluate_charge, *, filling: float, lower: float, upper: float) -> tuple[float, float]:
    lower_charge, _, _ = evaluate_charge(lower)
    upper_charge, _, _ = evaluate_charge(upper)
    while lower_charge > filling or upper_charge < filling:
        lower *= 2.0
        upper *= 2.0
        lower_charge, _, _ = evaluate_charge(lower)
        upper_charge, _, _ = evaluate_charge(upper)
    return lower, upper


def _solve_mu(
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

    raise ValueError("Chemical-potential solver did not converge within max_mu_iterations")


def _density_from_matrix(matrix: np.ndarray, keys: list[tuple[int, ...]]) -> tuple[_tb_type, _tb_type]:
    zero_key = tuple(0 for _ in next(iter(keys), tuple()))
    rho = {key: matrix if key == zero_key else np.zeros_like(matrix) for key in keys}
    error = {key: np.zeros_like(matrix) for key in keys}
    return rho, error


def _density_matrix_at_mu_zero_dim(
    matrix: np.ndarray,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
) -> tuple[_tb_type, _tb_type, DensityIntegrationInfo]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    occupation = fermi_dirac(eigenvalues, kT, mu)
    density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
    rho, error = _density_from_matrix(density, keys)
    info = DensityIntegrationInfo(
        n_kernel_evals=1,
        n_evaluator_evals=1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
    )
    return rho, error, info


def _density_matrix_zero_dim(
    matrix: np.ndarray,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    mu_guess: float,
    charge_tol: float,
    mu_xtol: float,
    max_mu_iterations: int,
    density_atol: float,
    density_rtol: float,
) -> tuple[_tb_type, _tb_type, float, FixedFillingInfo]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if kT == 0:
        mu, charge = _zero_dim_zero_temp_mu(eigenvalues, filling=filling, mu_guess=mu_guess)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
        rho, error = _density_from_matrix(density, keys)
        info = FixedFillingInfo(
            mu=mu,
            charge=charge,
            charge_error=abs(charge - filling),
            dcharge_dmu=0.0,
            root_iterations=1,
            charge_integration_calls=1,
            density_integration_calls=1,
            charge_n_kernel_evals=1,
            density_n_kernel_evals=0,
            n_kernel_evals=1,
            charge_n_evaluator_evals=1,
            density_n_evaluator_evals=1,
            n_evaluator_evals=2,
            n_cached_nodes=1,
            n_leaves=1,
            n_leaf_nodes=1,
            subdivisions=0,
            charge_integral_atol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
        )
        return rho, error, mu, info

    lower, upper = _mu_bracket({tuple(): matrix}, kT)
    charge_calls = 0

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        nonlocal charge_calls
        charge_calls += 1
        occupation = fermi_dirac(eigenvalues, kT, mu)
        charge = float(np.sum(occupation))
        derivative = float(np.sum(occupation * (1.0 - occupation) / kT))
        return charge, 0.0, derivative

    lower, upper = _expand_mu_bracket(
        evaluate_charge,
        filling=filling,
        lower=lower,
        upper=upper,
    )
    mu, charge, charge_error, derivative, iteration = _solve_mu(
        evaluate_charge,
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=charge_tol,
        mu_xtol=mu_xtol,
        max_mu_iterations=max_mu_iterations,
    )

    occupation = fermi_dirac(eigenvalues, kT, mu)
    density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
    rho, error = _density_from_matrix(density, keys)
    charge_integral_atol, _ = _charge_integral_tolerance(charge_tol)
    info = FixedFillingInfo(
        mu=mu,
        charge=charge,
        charge_error=charge_error,
        dcharge_dmu=derivative,
        root_iterations=iteration,
        charge_integration_calls=charge_calls,
        density_integration_calls=1,
        charge_n_kernel_evals=1,
        density_n_kernel_evals=0,
        n_kernel_evals=1,
        charge_n_evaluator_evals=charge_calls,
        density_n_evaluator_evals=1,
        n_evaluator_evals=charge_calls + 1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=density_rtol,
    )
    return rho, error, mu, info


def density_matrix_at_mu(
    h: _tb_type,
    mu: float,
    kT: float,
    keys: list,
    density_atol: float = 1e-6,
    density_rtol: float = 0.0,
    max_subdivisions: int | None = 50_000,
    rule: str = "auto",
    batch_size: int | None = None,
) -> tuple[_tb_type, _tb_type, DensityIntegrationInfo]:
    """Compute the real-space density matrix at a fixed chemical potential."""
    if kT < 0:
        raise ValueError("density_matrix_at_mu requires kT >= 0")

    keys = _normalize_keys(h, keys)
    ndim = len(next(iter(h)))
    if ndim == 0:
        if set(h) != {tuple()}:
            raise ValueError("Zero-dimensional evaluation expects only the local key")
        return _density_matrix_at_mu_zero_dim(h[tuple()], mu=mu, kT=kT, keys=keys)
    if kT == 0:
        if rule != "auto" or batch_size is not None:
            raise ValueError("rule and batch_size are supported only for kT > 0")
        from meanfi.zero_temp import density_matrix_at_mu_zero_temp

        rho, error, info = density_matrix_at_mu_zero_temp(
            h,
            mu=mu,
            keys=keys,
            density_atol=density_atol,
            density_rtol=density_rtol,
            max_subdivisions=max_subdivisions,
        )
        return rho, error, info

    ndof = next(iter(h.values())).shape[0]
    integrator = _build_integrator(
        h,
        evaluator=_density_evaluator(ndim, ndof, keys, kT),
        rule=rule,
        batch_size=batch_size,
    )
    result = _run_integrator(
        integrator,
        mu,
        atol=density_atol,
        rtol=density_rtol,
        max_subdivisions=max_subdivisions,
        error_message="Adaptive quadrature did not converge",
    )
    rho, error = _split_density_result(result.estimate, result.error, ndof, keys)
    return rho, error, _integration_stats(result)


def density_matrix(
    h: _tb_type,
    filling: float,
    kT: float,
    keys: list,
    charge_tol: float = 1e-6,
    density_atol: float = 1e-6,
    density_rtol: float = 0.0,
    mu_guess: float = 0.0,
    mu_xtol: float = 1e-6,
    max_mu_iterations: int = 64,
    max_subdivisions: int | None = 50_000,
    rule: str = "auto",
    batch_size: int | None = None,
) -> tuple[_tb_type, _tb_type, float, FixedFillingInfo]:
    """Compute the fixed-filling real-space density matrix with adaptive quadrature."""
    if kT < 0:
        raise ValueError("density_matrix requires kT >= 0")

    keys = _normalize_keys(h, keys)
    ndim = len(next(iter(h)))
    if ndim == 0:
        if set(h) != {tuple()}:
            raise ValueError("Zero-dimensional evaluation expects only the local key")
        return _density_matrix_zero_dim(
            h[tuple()],
            filling=filling,
            kT=kT,
            keys=keys,
            mu_guess=mu_guess,
            charge_tol=charge_tol,
            mu_xtol=mu_xtol,
            max_mu_iterations=max_mu_iterations,
            density_atol=density_atol,
            density_rtol=density_rtol,
        )
    if kT == 0:
        if rule != "auto" or batch_size is not None:
            raise ValueError("rule and batch_size are supported only for kT > 0")
        from meanfi.zero_temp import density_matrix_zero_temp

        rho, error, mu, info = density_matrix_zero_temp(
            h,
            filling=filling,
            keys=keys,
            charge_tol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            mu_guess=mu_guess,
            mu_xtol=mu_xtol,
            max_mu_iterations=max_mu_iterations,
            max_subdivisions=max_subdivisions,
        )
        return rho, error, mu, info

    ndof = next(iter(h.values())).shape[0]
    charge_integral_atol, charge_integral_rtol = _charge_integral_tolerance(charge_tol)
    charge_integrator = _build_integrator(
        h,
        evaluator=_charge_evaluator(ndim, ndof, kT),
        rule=rule,
        batch_size=batch_size,
    )

    charge_integration_calls = 0
    charge_kernel_evals = 0
    charge_evaluator_evals = 0

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        nonlocal charge_integration_calls, charge_kernel_evals, charge_evaluator_evals
        result = _run_integrator(
            charge_integrator,
            mu,
            atol=charge_integral_atol,
            rtol=charge_integral_rtol,
            max_subdivisions=max_subdivisions,
            error_message=(
                "Adaptive quadrature did not converge while solving for the chemical potential"
            ),
        )
        charge, charge_error, derivative = _split_charge_result(result.estimate, result.error)
        charge_integration_calls += 1
        charge_kernel_evals += int(result.n_kernel_evals)
        charge_evaluator_evals += int(result.n_evaluator_evals)
        return charge, charge_error, derivative

    lower, upper = _mu_bracket(h, kT)
    lower, upper = _expand_mu_bracket(
        evaluate_charge,
        filling=filling,
        lower=lower,
        upper=upper,
    )
    mu, charge, charge_error, derivative, iteration = _solve_mu(
        evaluate_charge,
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=charge_tol,
        mu_xtol=mu_xtol,
        max_mu_iterations=max_mu_iterations,
    )

    density_integrator = charge_integrator.replace_evaluator(
        _density_evaluator(ndim, ndof, keys, kT)
    )
    density_result = _run_integrator(
        density_integrator,
        mu,
        atol=density_atol,
        rtol=density_rtol,
        max_subdivisions=max_subdivisions,
        error_message="Adaptive quadrature did not converge while evaluating density",
    )

    rho, error = _split_density_result(density_result.estimate, density_result.error, ndof, keys)
    density_stats = _integration_stats(density_result)
    info = FixedFillingInfo(
        mu=mu,
        charge=charge,
        charge_error=charge_error,
        dcharge_dmu=derivative,
        root_iterations=iteration,
        charge_integration_calls=charge_integration_calls,
        density_integration_calls=1,
        charge_n_kernel_evals=charge_kernel_evals,
        density_n_kernel_evals=int(density_result.n_kernel_evals),
        n_kernel_evals=charge_kernel_evals + int(density_result.n_kernel_evals),
        charge_n_evaluator_evals=charge_evaluator_evals,
        density_n_evaluator_evals=int(density_result.n_evaluator_evals),
        n_evaluator_evals=charge_evaluator_evals + int(density_result.n_evaluator_evals),
        n_cached_nodes=density_stats.n_cached_nodes,
        n_leaves=density_stats.n_leaves,
        n_leaf_nodes=density_stats.n_leaf_nodes,
        subdivisions=density_stats.subdivisions,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=density_rtol,
    )
    return rho, error, mu, info


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the mean-field correction from the density matrix."""
    n = len(list(density_matrix)[0])
    local_key = tuple(np.zeros((n,), dtype=int))

    direct = {
        local_key: np.sum(
            np.array(
                [
                    np.diag(
                        np.einsum("pp,pn->n", density_matrix[local_key], h_int[vec])
                    )
                    for vec in frozenset(h_int)
                ]
            ),
            axis=0,
        )
    }

    exchange = {
        vec: -1 * h_int.get(vec, 0) * density_matrix[vec] for vec in frozenset(h_int)
    }
    return add_tb(direct, exchange)


def _zero_dim_zero_temp_mu(
    eigenvalues: np.ndarray,
    *,
    filling: float,
    mu_guess: float,
) -> tuple[float, float]:
    unique = np.unique(np.asarray(eigenvalues, dtype=float))
    candidates: list[tuple[float, bool]] = []
    if unique.size == 0:
        return 0.0, 0.0

    candidates.append((unique[0] - 1.0, False))
    for energy in unique:
        candidates.append((float(energy), False))
    for left, right in zip(unique[:-1], unique[1:], strict=False):
        candidates.append((0.5 * float(left + right), True))
    candidates.append((unique[-1] + 1.0, False))

    best_mu = float(mu_guess)
    best_charge = float(np.sum(fermi_dirac(eigenvalues, 0.0, best_mu)))
    best_key = (abs(best_charge - filling), 1, abs(best_mu - mu_guess))
    for candidate_mu, is_midgap in candidates:
        charge = float(np.sum(fermi_dirac(eigenvalues, 0.0, candidate_mu)))
        candidate_key = (
            abs(charge - filling),
            0 if is_midgap else 1,
            abs(candidate_mu - mu_guess),
        )
        if candidate_key < best_key:
            best_key = candidate_key
            best_mu = float(candidate_mu)
            best_charge = charge
    return best_mu, best_charge
