from __future__ import annotations

import numpy as np

from meanfi.core.results import DensityMatrixResult, FixedFillingInfo
from meanfi.integrate.common import uniform_grid_info, wrap_adaptive_result, wrap_density_result
from meanfi.integrate.density_support import (
    DensityEntrySupport,
    full_density_entry_support,
    workspace_complex_dtype,
)
from meanfi.integrate.fixed_filling import solve_fixed_filling_root
from meanfi.integrate.matrix_functions import (
    BdGMatrixFunction,
    DirectDiagonalization,
    basis_block,
    density_block,
    RationalFOE,
    resolve_matrix_function,
    shift_by_mu,
)
from meanfi.integrate.matrix_functions.rational import PreparedRationalNode
from meanfi.integrate.methods import AdaptiveQuadrature, IntegrationMethod, UniformGrid
from meanfi.integrate.quadrature.bdg_backend import build_bdg_backend
from meanfi.core.matrix import is_sparse_like
from meanfi.integrate.quadrature.runtime import solve_quadrature_fixed_filling
from meanfi.superconducting.bdg import charge_diagonal, mu_bracket_for_bdg
from meanfi.tb.tb import _tb_type
from meanfi.integrate.uniform_grid import (
    build_uniform_grid_node_bundle,
    resolve_uniform_grid_matrix_function,
    uniform_grid_fixed_filling_from_nodes,
)


def effective_bdg_filling_tol(
    *,
    filling_tol: float | None,
    density_matrix_tol: float,
    filling_weights: np.ndarray,
) -> float:
    if filling_tol is not None:
        if filling_tol <= 0:
            raise ValueError("filling_tol must be positive when provided")
        return float(filling_tol)
    return float(np.sum(np.abs(filling_weights)) * density_matrix_tol)


def matrix_function(integration: IntegrationMethod, hamiltonian: _tb_type, *, kT: float) -> BdGMatrixFunction:
    selected = getattr(integration, "matrix_function", None)
    if isinstance(integration, UniformGrid):
        return resolve_uniform_grid_matrix_function(selected, hamiltonian, kT=kT)
    if selected is None and any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
        return RationalFOE(rational_scheme="aaa")
    return resolve_matrix_function(selected)


def _local_filling(block: np.ndarray, indices, weights: np.ndarray) -> float:
    values = block[np.asarray(indices, dtype=int), np.arange(len(indices))]
    return float(np.real(np.sum(weights * values)))


def _solve_bdg_zero_dim(
    *,
    hamiltonian: _tb_type,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
    q_diag: np.ndarray,
    selected_matrix_function: BdGMatrixFunction,
    filling_indices,
    filling_weights: np.ndarray,
    density_entry_support: DensityEntrySupport,
) -> DensityMatrixResult:
    from meanfi.superconducting.bdg import mu_bracket_for_bdg

    workspace_dtype = workspace_complex_dtype(integration)
    matrix = hamiltonian[tuple()]
    filling_block = basis_block(q_diag.size, filling_indices, dtype=workspace_dtype)
    trace_weights = np.zeros(q_diag.size, dtype=float)
    trace_weights[np.asarray(filling_indices, dtype=int)] = np.asarray(
        filling_weights,
        dtype=float,
    )
    prepared_node = None
    if isinstance(selected_matrix_function, RationalFOE):
        prepared_node = PreparedRationalNode(
            matrix,
            kT=kT,
            q_diag=q_diag,
            options=selected_matrix_function,
            charge_tolerance=filling_tol,
            workspace_dtype=workspace_dtype,
            trace_weights_diag=trace_weights,
        )

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        if prepared_node is not None:
            charge, derivative = prepared_node.charge_and_derivative(mu)
            return charge, 0.0, derivative
        result = density_block(
            selected_matrix_function,
            shift_by_mu(matrix, mu, q_diag, dtype=workspace_dtype),
            filling_block,
            kT=kT,
            q_diag=q_diag,
            derivative=True,
            tolerance=integration.density_matrix_tol,
            derivative_trace_monitor=lambda block: _local_filling(
                block,
                filling_indices,
                filling_weights,
            ),
            derivative_context=f"mu={float(mu):.12g}",
            workspace_dtype=workspace_dtype,
        )
        derivative_block = result.derivative_block
        derivative = (
            0.0
            if derivative_block is None
            else _local_filling(derivative_block, filling_indices, filling_weights)
        )
        return _local_filling(result.block, filling_indices, filling_weights), 0.0, derivative

    root = solve_fixed_filling_root(
        evaluate_charge=evaluate_charge,
        mu_bracket=lambda: mu_bracket_for_bdg(hamiltonian, kT),
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
    )

    density_basis = density_entry_support.basis_block(dtype=workspace_dtype)
    if isinstance(selected_matrix_function, DirectDiagonalization):
        density_basis = np.eye(q_diag.size, dtype=workspace_dtype)
        density = density_block(
            selected_matrix_function,
            shift_by_mu(matrix, root.mu, q_diag, dtype=workspace_dtype),
            density_basis,
            kT=kT,
            q_diag=q_diag,
            derivative=False,
            tolerance=integration.density_matrix_tol,
            workspace_dtype=workspace_dtype,
        ).block
        density_matrix = {key: density if key == tuple() else np.zeros_like(density) for key in keys}
        density_matrix_error = {key: np.zeros_like(density, dtype=float) for key in keys}
    else:
        assert prepared_node is not None
        density = prepared_node.density_columns_from_charge_order(root.mu, density_basis)
        density_matrix, density_matrix_error = density_entry_support.expand_entries(
            density_entry_support.pack_columns(density),
            np.zeros(density_entry_support.output_size, dtype=float),
        )
    raw_info = FixedFillingInfo(
        mu=root.mu,
        charge=root.charge,
        charge_error=root.charge_error,
        dcharge_dmu=float("nan") if root.derivative is None else root.derivative,
        root_iterations=root.root_iterations,
        charge_integration_calls=root.root_iterations,
        density_integration_calls=1,
        charge_n_kernel_evals=1,
        density_n_kernel_evals=1,
        n_kernel_evals=1,
        unique_evals=1,
        charge_n_evaluator_evals=root.root_iterations,
        density_n_evaluator_evals=1,
        n_evaluator_evals=root.root_iterations + 1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
        charge_integral_atol=filling_tol,
        density_atol=integration.density_matrix_tol,
        density_rtol=0.0,
        error_estimate_available=True,
    )
    if isinstance(integration, UniformGrid):
        return wrap_density_result(
            density_matrix=density_matrix,
            density_matrix_error=None,
            mu=raw_info.mu,
            filling=raw_info.charge,
            target_filling=filling,
            integration=integration,
            info=uniform_grid_info(
                integration=integration,
                hamiltonian=hamiltonian,
                n_kernel_evals=raw_info.n_kernel_evals,
                n_evaluator_evals=raw_info.n_evaluator_evals,
                root_iterations=raw_info.root_iterations,
                charge_integration_calls=raw_info.charge_integration_calls,
                density_integration_calls=raw_info.density_integration_calls,
                error_estimate_available=False,
            ),
            keys=keys,
        )
    return wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
        integration=integration,
        keys=keys,
    )


def solve_bdg_density_fixed_filling(
    model,
    meanfield: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
    density_entry_support: DensityEntrySupport | None = None,
) -> DensityMatrixResult:
    if model.kT <= 0:
        raise NotImplementedError("BdG density currently requires kT > 0")
    if mu_tol <= 0:
        raise ValueError("mu_tol must be positive")
    if max_mu_iterations is not None and max_mu_iterations <= 0:
        raise ValueError("max_mu_iterations must be positive")

    hamiltonian = model.bdg_hamiltonian_from_meanfield(meanfield)
    selected_matrix_function = matrix_function(integration, hamiltonian, kT=model.kT)
    q_diag = charge_diagonal(model._ndof)
    filling_indices = tuple(range(model._ndof))
    filling_weights = np.ones(model._ndof, dtype=float)
    resolved_filling_tol = effective_bdg_filling_tol(
        filling_tol=filling_tol,
        density_matrix_tol=getattr(integration, "density_matrix_tol"),
        filling_weights=filling_weights,
    )
    density_support = density_entry_support

    if model._ndim == 0:
        return _solve_bdg_zero_dim(
            hamiltonian=hamiltonian,
            filling=model.filling,
            kT=model.kT,
            keys=keys,
            integration=integration,
            filling_tol=resolved_filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_guess=mu_guess,
            q_diag=q_diag,
            selected_matrix_function=selected_matrix_function,
            filling_indices=filling_indices,
            filling_weights=filling_weights,
            density_entry_support=density_support,
        )

    if isinstance(integration, UniformGrid):
        workspace_dtype = workspace_complex_dtype(integration)
        resolved_density_support = (
            density_support
            if density_support is not None
            else full_density_entry_support(keys, size=2 * model._ndof)
        )
        bundle = build_uniform_grid_node_bundle(
            hamiltonian,
            kT=model.kT,
            nk=integration.nk,
            keys=keys,
            matrix_function=selected_matrix_function,
            q_diag=q_diag,
            trace_weights_diag=np.concatenate(
                [np.ones(model._ndof, dtype=float), np.zeros(model._ndof, dtype=float)]
            ),
            charge_tolerance=resolved_filling_tol,
            density_tolerance=integration.density_matrix_tol,
            density_entry_support=resolved_density_support,
            workspace_dtype=workspace_dtype,
        )
        density_matrix, mu, resolved_filling, info = uniform_grid_fixed_filling_from_nodes(
            bundle,
            hamiltonian=hamiltonian,
            integration=integration,
            filling=model.filling,
            mu_guess=mu_guess,
            filling_tol=resolved_filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_bracket_builder=lambda: mu_bracket_for_bdg(hamiltonian, model.kT),
        )
        return wrap_density_result(
            density_matrix=density_matrix,
            density_matrix_error=None,
            mu=mu,
            filling=resolved_filling,
            target_filling=model.filling,
            integration=integration,
            info=uniform_grid_info(
                integration=integration,
                hamiltonian=hamiltonian,
                n_kernel_evals=info.n_kernel_evals,
                n_evaluator_evals=info.n_evaluator_evals,
                root_iterations=info.root_iterations,
                charge_integration_calls=info.charge_integration_calls,
                density_integration_calls=info.density_integration_calls,
                error_estimate_available=False,
            ),
            keys=keys,
        )

    if not isinstance(integration, AdaptiveQuadrature):
        raise NotImplementedError("BdG density currently requires AdaptiveQuadrature or UniformGrid")

    backend = build_bdg_backend(
        hamiltonian,
        keys=keys,
        kT=model.kT,
        q_diag=q_diag,
        matrix_function=selected_matrix_function,
        filling_indices=filling_indices,
        filling_weights=filling_weights,
        tolerance=integration.density_matrix_tol,
        charge_tolerance=resolved_filling_tol,
        density_entry_support=density_support,
        workspace_dtype=workspace_complex_dtype(integration),
    )
    density_matrix, density_matrix_error, raw_info = solve_quadrature_fixed_filling(
        backend,
        filling=model.filling,
        mu_guess=mu_guess,
        rule=integration.rule,
        batch_size=integration.batch_size,
        filling_tol=resolved_filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        density_atol=integration.density_matrix_tol,
        max_subdivisions=integration.max_refinements,
        root_error_message=(
            "BdG adaptive quadrature did not converge while solving for the chemical potential"
        ),
        density_error_message="BdG adaptive quadrature did not converge while evaluating density",
    )
    return wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=model.filling,
        integration=integration,
        keys=keys,
    )
