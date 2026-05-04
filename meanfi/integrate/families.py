from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from meanfi.core.validation import require_zero_dim_local_key_only, tb_dimension, zero_key
from meanfi.tb.tb import _tb_type

from .common import effective_filling_tol, local_density_filling, retarget_result_keys, wrap_adaptive_result
from .density_support import DensityEntrySupport, workspace_complex_dtype
from .methods import AdaptiveQuadrature, AdaptiveSimplex, IntegrationMethod, UniformGrid
from .quadrature.normal_backend import build_normal_backend, resolve_normal_matrix_function
from .quadrature.runtime import solve_quadrature_at_mu, solve_quadrature_fixed_filling
from .simplex import density_matrix_at_mu_zero_temp, density_matrix_zero_temp
from .uniform_grid import solve_uniform_grid_at_mu, solve_uniform_grid_fixed_filling
from .zero_dim import density_matrix_at_mu_zero_dim, density_matrix_zero_dim


@dataclass(frozen=True)
class DispatchContext:
    hamiltonian: _tb_type
    kT: float
    integration: IntegrationMethod
    requested_keys: list[tuple[int, ...]]
    solve_keys: list[tuple[int, ...]]
    density_entry_support: DensityEntrySupport | None = None


@dataclass(frozen=True)
class IntegrationHandler:
    solve_at_mu: Callable[[DispatchContext, float], object]
    solve_fixed_filling: Callable[
        [DispatchContext, float, float | None, float, int | None, float],
        object,
    ]


def _adaptive_quadrature_at_mu(context: DispatchContext, mu: float):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveQuadrature)
    resolve_normal_matrix_function(getattr(integration, "matrix_function", None), hamiltonian)

    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            hamiltonian[tuple()],
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
            density_entry_support=None,
            matrix_function=getattr(integration, "matrix_function", None),
            workspace_dtype=workspace_complex_dtype(integration),
            density_tolerance=integration.density_matrix_tol,
        )
    else:
        backend = build_normal_backend(
            hamiltonian,
            integration=integration,
            keys=context.solve_keys,
            kT=context.kT,
            density_entry_support=context.density_entry_support,
        )
        density_matrix, density_matrix_error, raw_info = solve_quadrature_at_mu(
            backend,
            mu=mu,
            rule=integration.rule,
            batch_size=integration.batch_size,
            density_atol=integration.density_matrix_tol,
            max_subdivisions=integration.max_refinements,
            error_message="Adaptive quadrature did not converge",
        )

    result = wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=mu,
        filling=local_density_filling(
            density_matrix,
            local_key=zero_key(tb_dimension(hamiltonian)),
        ),
        target_filling=None,
        integration=integration,
        keys=context.solve_keys,
    )
    return retarget_result_keys(result, keys=context.requested_keys)


def _adaptive_quadrature_fixed_filling(
    context: DispatchContext,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveQuadrature)

    resolved_filling_tol = effective_filling_tol(
        integration,
        hamiltonian=hamiltonian,
        filling_tol=filling_tol,
    )
    resolve_normal_matrix_function(getattr(integration, "matrix_function", None), hamiltonian)
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_dim(
            hamiltonian[tuple()],
            filling=filling,
            kT=context.kT,
            keys=context.solve_keys,
            mu_guess=mu_guess,
            charge_tol=resolved_filling_tol,
            mu_xtol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            density_entry_support=None,
            matrix_function=getattr(integration, "matrix_function", None),
            workspace_dtype=workspace_complex_dtype(integration),
        )
    else:
        backend = build_normal_backend(
            hamiltonian,
            integration=integration,
            keys=context.solve_keys,
            kT=context.kT,
            fixed_filling_tolerance=resolved_filling_tol,
            density_entry_support=context.density_entry_support,
        )
        density_matrix, density_matrix_error, raw_info = solve_quadrature_fixed_filling(
            backend,
            filling=filling,
            mu_guess=mu_guess,
            rule=integration.rule,
            batch_size=integration.batch_size,
            filling_tol=resolved_filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            density_atol=integration.density_matrix_tol,
            max_subdivisions=integration.max_refinements,
            root_error_message=(
                "Adaptive quadrature did not converge while solving for the chemical potential"
            ),
            density_error_message=(
                "Adaptive quadrature did not converge while evaluating density"
            ),
        )

    result = wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
        integration=integration,
        keys=context.solve_keys,
    )
    return retarget_result_keys(result, keys=context.requested_keys)


def _adaptive_simplex_at_mu(context: DispatchContext, mu: float):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveSimplex)

    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            hamiltonian[tuple()],
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
        )
    else:
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_temp(
            hamiltonian,
            mu=mu,
            keys=context.solve_keys,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            max_subdivisions=integration.max_refinements,
        )

    result = wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=mu,
        filling=local_density_filling(
            density_matrix,
            local_key=zero_key(tb_dimension(hamiltonian)),
        ),
        target_filling=None,
        integration=integration,
        keys=context.solve_keys,
    )
    return retarget_result_keys(result, keys=context.requested_keys)


def _adaptive_simplex_fixed_filling(
    context: DispatchContext,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveSimplex)

    resolved_filling_tol = effective_filling_tol(
        integration,
        hamiltonian=hamiltonian,
        filling_tol=filling_tol,
    )
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_dim(
            hamiltonian[tuple()],
            filling=filling,
            kT=context.kT,
            keys=context.solve_keys,
            mu_guess=mu_guess,
            charge_tol=resolved_filling_tol,
            mu_xtol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
        )
    else:
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_temp(
            hamiltonian,
            filling=filling,
            keys=context.solve_keys,
            charge_tol=resolved_filling_tol,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            mu_guess=mu_guess,
            mu_xtol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            max_subdivisions=integration.max_refinements,
        )

    result = wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
        integration=integration,
        keys=context.solve_keys,
    )
    return retarget_result_keys(result, keys=context.requested_keys)


def _uniform_grid_at_mu(context: DispatchContext, mu: float):
    integration = context.integration
    assert isinstance(integration, UniformGrid)
    return retarget_result_keys(
        solve_uniform_grid_at_mu(
            context.hamiltonian,
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
            integration=integration,
            density_entry_support=context.density_entry_support,
        ),
        keys=context.requested_keys,
    )


def _uniform_grid_fixed_filling(
    context: DispatchContext,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
):
    del mu_guess
    integration = context.integration
    assert isinstance(integration, UniformGrid)
    return retarget_result_keys(
        solve_uniform_grid_fixed_filling(
            context.hamiltonian,
            filling=filling,
            kT=context.kT,
            keys=context.solve_keys,
            integration=integration,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            density_entry_support=context.density_entry_support,
        ),
        keys=context.requested_keys,
    )


_HANDLERS: dict[type[IntegrationMethod], IntegrationHandler] = {
    AdaptiveQuadrature: IntegrationHandler(
        solve_at_mu=_adaptive_quadrature_at_mu,
        solve_fixed_filling=_adaptive_quadrature_fixed_filling,
    ),
    AdaptiveSimplex: IntegrationHandler(
        solve_at_mu=_adaptive_simplex_at_mu,
        solve_fixed_filling=_adaptive_simplex_fixed_filling,
    ),
    UniformGrid: IntegrationHandler(
        solve_at_mu=_uniform_grid_at_mu,
        solve_fixed_filling=_uniform_grid_fixed_filling,
    ),
}


def integration_handler(integration: IntegrationMethod) -> IntegrationHandler:
    for integration_type, handler in _HANDLERS.items():
        if isinstance(integration, integration_type):
            return handler
    raise TypeError("integration must be an IntegrationMethod instance")
