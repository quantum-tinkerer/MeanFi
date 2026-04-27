from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from meanfi.core.validation import require_zero_dim_local_key_only, tb_dimension, zero_key
from meanfi.tb.tb import _tb_type

from .common import (
    effective_filling_tol,
    local_density_filling,
    prepare_keys,
    retarget_result_keys,
    validate_integration_method,
    wrap_adaptive_result,
)
from .methods import AdaptiveQuadrature, AdaptiveSimplex, IntegrationMethod, UniformGrid
from .quadrature.normal_backend import build_normal_backend
from .quadrature.runtime import solve_quadrature_at_mu, solve_quadrature_fixed_filling
from .simplex import density_matrix_at_mu_zero_temp, density_matrix_zero_temp
from .uniform_grid import solve_uniform_grid_at_mu, solve_uniform_grid_fixed_filling
from .zero_dim import density_matrix_at_mu_zero_dim, density_matrix_zero_dim


@dataclass(frozen=True)
class _DispatchContext:
    hamiltonian: _tb_type
    kT: float
    integration: IntegrationMethod
    requested_keys: list[tuple[int, ...]]
    solve_keys: list[tuple[int, ...]]


@dataclass(frozen=True)
class _IntegrationHandler:
    solve_at_mu: Callable[[_DispatchContext, float], object]
    solve_fixed_filling: Callable[
        [_DispatchContext, float, float | None, float, int | None, float],
        object,
    ]


def _adaptive_quadrature_at_mu(context: _DispatchContext, mu: float):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveQuadrature)

    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            hamiltonian[tuple()],
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
        )
    else:
        backend = build_normal_backend(hamiltonian, keys=context.solve_keys, kT=context.kT)
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
    context: _DispatchContext,
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
        backend = build_normal_backend(hamiltonian, keys=context.solve_keys, kT=context.kT)
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


def _adaptive_simplex_at_mu(context: _DispatchContext, mu: float):
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
    context: _DispatchContext,
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


def _uniform_grid_at_mu(context: _DispatchContext, mu: float):
    integration = context.integration
    assert isinstance(integration, UniformGrid)
    return retarget_result_keys(
        solve_uniform_grid_at_mu(
            context.hamiltonian,
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
            integration=integration,
        ),
        keys=context.requested_keys,
    )


def _uniform_grid_fixed_filling(
    context: _DispatchContext,
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
        ),
        keys=context.requested_keys,
    )


_HANDLERS: dict[type[IntegrationMethod], _IntegrationHandler] = {
    AdaptiveQuadrature: _IntegrationHandler(
        solve_at_mu=_adaptive_quadrature_at_mu,
        solve_fixed_filling=_adaptive_quadrature_fixed_filling,
    ),
    AdaptiveSimplex: _IntegrationHandler(
        solve_at_mu=_adaptive_simplex_at_mu,
        solve_fixed_filling=_adaptive_simplex_fixed_filling,
    ),
    UniformGrid: _IntegrationHandler(
        solve_at_mu=_uniform_grid_at_mu,
        solve_fixed_filling=_uniform_grid_fixed_filling,
    ),
}


def _integration_handler(integration: IntegrationMethod) -> _IntegrationHandler:
    for integration_type, handler in _HANDLERS.items():
        if isinstance(integration, integration_type):
            return handler
    raise TypeError("integration must be an IntegrationMethod instance")


def solve_density_matrix_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
):
    validate_integration_method(integration, kT=kT)
    requested_keys, working_keys, _local_key = prepare_keys(hamiltonian, keys)
    context = _DispatchContext(
        hamiltonian=hamiltonian,
        kT=kT,
        integration=integration,
        requested_keys=requested_keys,
        solve_keys=requested_keys if working_keys == requested_keys else working_keys,
    )
    return _integration_handler(integration).solve_at_mu(context, mu)


def solve_density_matrix_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float = 0.0,
):
    validate_integration_method(integration, kT=kT)
    if mu_tol <= 0:
        raise ValueError("mu_tol must be positive")
    if max_mu_iterations is not None and max_mu_iterations <= 0:
        raise ValueError("max_mu_iterations must be positive")

    requested_keys, working_keys, _local_key = prepare_keys(hamiltonian, keys)
    context = _DispatchContext(
        hamiltonian=hamiltonian,
        kT=kT,
        integration=integration,
        requested_keys=requested_keys,
        solve_keys=requested_keys if working_keys == requested_keys else working_keys,
    )
    return _integration_handler(integration).solve_fixed_filling(
        context,
        filling,
        filling_tol,
        mu_tol,
        max_mu_iterations,
        mu_guess,
    )
