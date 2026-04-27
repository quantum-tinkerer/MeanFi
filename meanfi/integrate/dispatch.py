from __future__ import annotations

from meanfi.tb.tb import _tb_type

from .common import prepare_keys, validate_integration_method
from .families import DispatchContext, integration_handler
from .methods import IntegrationMethod


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
    context = DispatchContext(
        hamiltonian=hamiltonian,
        kT=kT,
        integration=integration,
        requested_keys=requested_keys,
        solve_keys=working_keys,
    )
    return integration_handler(integration).solve_at_mu(context, mu)


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
    context = DispatchContext(
        hamiltonian=hamiltonian,
        kT=kT,
        integration=integration,
        requested_keys=requested_keys,
        solve_keys=working_keys,
    )
    return integration_handler(integration).solve_fixed_filling(
        context,
        filling,
        filling_tol,
        mu_tol,
        max_mu_iterations,
        mu_guess,
    )
