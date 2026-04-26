from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from meanfi.core.filling import charge_integral_tolerance, expand_mu_bracket, solve_mu
from meanfi.core.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.tb.tb import _tb_type

if TYPE_CHECKING:
    from stateful_quadrature import StatefulIntegrator
else:
    StatefulIntegrator = Any


@dataclass(frozen=True)
class QuadratureBackend:
    bounds: tuple[list[float], list[float]]
    kernel: Callable[[np.ndarray], np.ndarray]
    charge_evaluator: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    density_evaluator: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    split_charge_result: Callable[[np.ndarray, np.ndarray], tuple[float, float, float]]
    split_density_result: Callable[[np.ndarray, np.ndarray], tuple[_tb_type, _tb_type]]
    density_info_builder: Callable[[Any], DensityIntegrationInfo]
    fixed_filling_info_builder: Callable[..., FixedFillingInfo]
    mu_bracket: Callable[[], tuple[float, float]]


def build_integrator(
    backend: QuadratureBackend,
    *,
    evaluator,
    rule: str,
    batch_size: int | None,
):
    """Construct a stateful quadrature integrator for a backend."""

    try:
        from stateful_quadrature import StatefulIntegrator
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "Finite-temperature integration requires the optional stateful_quadrature dependency"
        ) from exc

    a, b = backend.bounds
    return StatefulIntegrator(
        a=a,
        b=b,
        kernel=backend.kernel,
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


def solve_quadrature_at_mu(
    backend: QuadratureBackend,
    *,
    mu: float,
    rule: str,
    batch_size: int | None,
    density_atol: float,
    max_subdivisions: int | None,
    error_message: str,
) -> tuple[_tb_type, _tb_type, DensityIntegrationInfo]:
    integrator = build_integrator(
        backend,
        evaluator=backend.density_evaluator,
        rule=rule,
        batch_size=batch_size,
    )
    result = run_integrator(
        integrator,
        mu,
        atol=density_atol,
        rtol=0.0,
        max_subdivisions=max_subdivisions,
        error_message=error_message,
    )
    density_matrix, density_matrix_error = backend.split_density_result(
        result.estimate,
        result.error,
    )
    return density_matrix, density_matrix_error, backend.density_info_builder(result)


def solve_quadrature_fixed_filling(
    backend: QuadratureBackend,
    *,
    filling: float,
    mu_guess: float,
    rule: str,
    batch_size: int | None,
    filling_tol: float,
    mu_tol: float,
    max_mu_iterations: int | None,
    density_atol: float,
    max_subdivisions: int | None,
    root_error_message: str,
    density_error_message: str,
) -> tuple[_tb_type, _tb_type, FixedFillingInfo]:
    charge_integral_atol, charge_integral_rtol = charge_integral_tolerance(filling_tol)
    charge_integrator = build_integrator(
        backend,
        evaluator=backend.charge_evaluator,
        rule=rule,
        batch_size=batch_size,
    )

    charge_integration_calls = 0
    charge_kernel_evals = 0
    charge_evaluator_evals = 0
    charge_subdivisions = 0

    def evaluate_charge(candidate_mu: float) -> tuple[float, float, float]:
        nonlocal charge_integration_calls, charge_kernel_evals, charge_evaluator_evals, charge_subdivisions
        result = run_integrator(
            charge_integrator,
            candidate_mu,
            atol=charge_integral_atol,
            rtol=charge_integral_rtol,
            max_subdivisions=max_subdivisions,
            error_message=root_error_message,
        )
        charge_integration_calls += 1
        charge_kernel_evals += int(result.n_kernel_evals)
        charge_evaluator_evals += int(result.n_evaluator_evals)
        charge_subdivisions += int(getattr(result, "subdivisions", 0))
        return backend.split_charge_result(result.estimate, result.error)

    lower, upper = backend.mu_bracket()
    lower, upper = expand_mu_bracket(
        evaluate_charge,
        filling=filling,
        lower=lower,
        upper=upper,
    )
    mu, resolved_filling, charge_error, derivative, iteration = solve_mu(
        evaluate_charge,
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=filling_tol,
        mu_xtol=mu_tol,
        max_mu_iterations=max_mu_iterations,
    )

    density_integrator = charge_integrator.replace_evaluator(backend.density_evaluator)
    density_result = run_integrator(
        density_integrator,
        mu,
        atol=density_atol,
        rtol=0.0,
        max_subdivisions=max_subdivisions,
        error_message=density_error_message,
    )
    density_matrix, density_matrix_error = backend.split_density_result(
        density_result.estimate,
        density_result.error,
    )
    raw_info = backend.fixed_filling_info_builder(
        mu=mu,
        charge=resolved_filling,
        charge_error=charge_error,
        derivative=derivative,
        root_iterations=iteration,
        charge_integration_calls=charge_integration_calls,
        charge_kernel_evals=charge_kernel_evals,
        charge_evaluator_evals=charge_evaluator_evals,
        charge_subdivisions=charge_subdivisions,
        density_result=density_result,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=0.0,
    )
    return density_matrix, density_matrix_error, raw_info
