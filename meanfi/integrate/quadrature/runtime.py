from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from stateful_quadrature import StatefulIntegrator

from meanfi.integrate.filling import charge_integral_tolerance, solve_mu
from meanfi.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.tb.ops import _tb_type


@dataclass(frozen=True)
class QuadratureBackend:
    bounds: tuple[list[float], list[float]]
    kernel: Callable[[np.ndarray], np.ndarray]
    payload_builder: Callable[[np.ndarray, np.ndarray], Any] | None
    charge_evaluator: Callable[[np.ndarray, Any, float], np.ndarray]
    density_evaluator: Callable[[np.ndarray, Any, float], np.ndarray]
    split_charge_result: Callable[
        [np.ndarray, np.ndarray], tuple[float, float, float | None]
    ]
    split_density_result: Callable[[np.ndarray, np.ndarray], tuple[_tb_type, _tb_type]]
    density_info_builder: Callable[[Any], DensityIntegrationInfo]
    fixed_filling_info_builder: Callable[..., FixedFillingInfo]
    mu_bracket: Callable[[], tuple[float, float]]
    freeze_density_mesh: bool = False
    charge_has_derivative: bool = True


def build_integrator(
    backend: QuadratureBackend,
    *,
    evaluator,
    rule: str,
    batch_size: int | None,
):
    """Construct a stateful quadrature integrator for a backend."""

    a, b = backend.bounds
    kwargs = dict(
        a=a,
        b=b,
        kernel=backend.kernel,
        evaluator=evaluator,
        rule=rule,
        batch_size=batch_size,
    )
    if backend.payload_builder is not None:
        kwargs["payload_builder"] = backend.payload_builder
    return StatefulIntegrator(**kwargs)


def run_integrator(
    integrator: StatefulIntegrator,
    parameter: float,
    *,
    atol: float,
    rtol: float,
    max_subdivisions: int | None,
    error_message: str,
    accepted_statuses: tuple[str, ...] = ("converged",),
):
    """Run the adaptive integrator and normalize non-convergence errors."""

    result = integrator.integrate(
        parameter,
        atol=atol,
        rtol=rtol,
        max_subdivisions=max_subdivisions,
    )
    if result.status not in accepted_statuses:
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
        accepted_statuses=("converged",),
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
    max_charge_evaluations: int | None,
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

    def evaluate_charge(candidate_mu: float) -> tuple[float, float, float | None]:
        nonlocal \
            charge_integration_calls, \
            charge_kernel_evals, \
            charge_evaluator_evals, \
            charge_subdivisions
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

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=backend.mu_bracket,
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        use_derivative=backend.charge_has_derivative,
    )

    density_integrator = charge_integrator.replace_evaluator(backend.density_evaluator)
    density_max_subdivisions = 0 if backend.freeze_density_mesh else max_subdivisions
    density_statuses = (
        ("converged", "max_subdivisions")
        if backend.freeze_density_mesh
        else ("converged",)
    )
    density_result = run_integrator(
        density_integrator,
        root.mu,
        atol=density_atol,
        rtol=0.0,
        max_subdivisions=density_max_subdivisions,
        error_message=density_error_message,
        accepted_statuses=density_statuses,
    )
    density_matrix, density_matrix_error = backend.split_density_result(
        density_result.estimate,
        density_result.error,
    )
    raw_info = backend.fixed_filling_info_builder(
        mu=root.mu,
        charge=root.charge,
        charge_error=root.charge_error,
        derivative=float("nan") if root.derivative is None else root.derivative,
        charge_evaluations=root.charge_evaluations,
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
