from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from stateful_quadrature import StatefulIntegrator

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


def density_integration_info(result) -> DensityIntegrationInfo:
    """Convert stateful-integrator metadata to internal density stats."""

    cached_nodes = getattr(result, "n_cached_nodes", getattr(result, "n_leaf_nodes", 0))
    return DensityIntegrationInfo(
        n_kernel_evals=int(result.n_kernel_evals),
        unique_evals=int(result.n_kernel_evals),
        n_evaluator_evals=int(result.n_evaluator_evals),
        n_cached_nodes=int(cached_nodes),
        n_leaves=int(getattr(result, "n_leaves", 0)),
        n_leaf_nodes=int(getattr(result, "n_leaf_nodes", cached_nodes)),
        subdivisions=int(getattr(result, "subdivisions", 0)),
        error_estimate_available=True,
    )


def fixed_filling_info(
    *,
    mu: float,
    charge: float,
    charge_error: float,
    derivative: float,
    charge_evaluations: int,
    charge_integration_calls: int,
    charge_kernel_evals: int,
    charge_evaluator_evals: int,
    charge_subdivisions: int,
    density_result,
    charge_integral_atol: float,
    density_atol: float,
    density_rtol: float,
) -> FixedFillingInfo:
    """Combine charge-root and density-integration stats."""

    density_stats = density_integration_info(density_result)
    density_kernel_evals = int(density_result.n_kernel_evals)
    density_evaluator_evals = int(density_result.n_evaluator_evals)
    return FixedFillingInfo(
        mu=mu,
        charge=charge,
        charge_error=charge_error,
        dcharge_dmu=derivative,
        charge_evaluations=charge_evaluations,
        charge_integration_calls=charge_integration_calls,
        density_integration_calls=1,
        charge_n_kernel_evals=charge_kernel_evals,
        density_n_kernel_evals=density_kernel_evals,
        n_kernel_evals=charge_kernel_evals + density_kernel_evals,
        unique_evals=charge_kernel_evals + density_kernel_evals,
        charge_n_evaluator_evals=charge_evaluator_evals,
        density_n_evaluator_evals=density_evaluator_evals,
        n_evaluator_evals=charge_evaluator_evals + density_evaluator_evals,
        n_cached_nodes=density_stats.n_cached_nodes,
        n_leaves=density_stats.n_leaves,
        n_leaf_nodes=density_stats.n_leaf_nodes,
        subdivisions=charge_subdivisions + density_stats.subdivisions,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=density_rtol,
        error_estimate_available=True,
    )


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
