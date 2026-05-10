"""Central density solve pipeline.

The public density APIs should read as one path: construct a physical problem,
build a numerical plan, evaluate at a chemical potential or solve fixed filling,
and return the public result object. Algorithm-specific modules still provide the
low-level matrix-function, quadrature, simplex, and uniform-grid evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.density.integrate.common import (
    uniform_grid_info,
    validate_integration_method,
    wrap_adaptive_result,
    wrap_density_result,
)
from meanfi.density.filling import solve_mu
from meanfi.density.kpoint.matrix_functions import (
    BdGMatrixFunction,
    DirectDiagonalization,
    RationalFOE,
    basis_block,
    density_block,
    resolve_sparse_default_matrix_function,
    shift_by_mu,
)
from meanfi.density.kpoint.matrix_functions.rational import PreparedMumpsRationalNode
from meanfi.density.integrate.methods import (
    AdaptiveQuadrature,
    IntegrationMethod,
    UniformGrid,
)
from meanfi.density.integrate.quadrature.bdg import build_bdg_backend
from meanfi.density.integrate.uniform import (
    build_uniform_grid_node_bundle,
    resolve_uniform_grid_matrix_function,
)
from meanfi.density.integrate.normal import (
    _solve_quadrature_fixed_filling,
    _uniform_fixed_filling_from_nodes,
)
from meanfi.density.filling import charge_diagonal, mu_bracket_for_bdg
from meanfi.density.integrate.workspace import workspace_complex_dtype
from meanfi.results import DensityMatrixResult, FixedFillingInfo
from meanfi.space.density_selection import DensitySelection
from meanfi.space.density_selection import full_density_selection
from meanfi.tb.ops import _tb_type, is_sparse_like


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
    return float(0.1 * np.sum(np.abs(filling_weights)) * density_matrix_tol)


def resolve_bdg_matrix_function(
    integration: IntegrationMethod,
    hamiltonian: _tb_type,
    *,
    kT: float,
) -> BdGMatrixFunction:
    selected = getattr(integration, "matrix_function", None)
    if isinstance(integration, UniformGrid):
        return resolve_uniform_grid_matrix_function(selected, hamiltonian, kT=kT)
    return resolve_sparse_default_matrix_function(
        selected,
        hamiltonian,
        parameter_name=f"{integration.__class__.__name__}.matrix_function",
    )


def _local_filling(block: np.ndarray, indices, weights: np.ndarray) -> float:
    values = block[np.asarray(indices, dtype=int), np.arange(len(indices))]
    return float(np.real(np.sum(weights * values)))


def _trace_weights(
    size: int,
    filling_indices,
    filling_weights: np.ndarray,
) -> np.ndarray:
    trace_weights = np.zeros(size, dtype=float)
    trace_weights[np.asarray(filling_indices, dtype=int)] = np.asarray(
        filling_weights,
        dtype=float,
    )
    return trace_weights


def _bdg_zero_dim_prepared_node(
    *,
    matrix,
    kT: float,
    q_diag: np.ndarray,
    selected_matrix_function: BdGMatrixFunction,
    filling_tol: float,
    density_tolerance: float,
    filling_indices,
    filling_weights: np.ndarray,
    density_selection: DensitySelection,
    workspace_dtype: np.dtype,
):
    if not isinstance(selected_matrix_function, RationalFOE):
        return None
    if not is_sparse_like(matrix):
        raise ValueError(
            "Dense RationalFOE is unsupported; use DirectDiagonalization for dense matrices "
            "or sparse matrices for the MUMPS selected-inverse RationalFOE path"
        )
    return PreparedMumpsRationalNode(
        matrix,
        kT=kT,
        q_diag=q_diag,
        options=selected_matrix_function,
        charge_tolerance=filling_tol,
        density_selection=density_selection,
        density_tolerance=density_tolerance,
        workspace_dtype=workspace_dtype,
        trace_weights_diag=_trace_weights(
            q_diag.size, filling_indices, filling_weights
        ),
    )


def _bdg_zero_dim_charge_evaluator(
    *,
    matrix,
    kT: float,
    q_diag: np.ndarray,
    selected_matrix_function: BdGMatrixFunction,
    filling_indices,
    filling_weights: np.ndarray,
    filling_block: np.ndarray,
    integration: IntegrationMethod,
    prepared_node,
    workspace_dtype: np.dtype,
):
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
        return (
            _local_filling(result.block, filling_indices, filling_weights),
            0.0,
            derivative,
        )

    return evaluate_charge


def _bdg_zero_dim_density(
    *,
    matrix,
    kT: float,
    root,
    q_diag: np.ndarray,
    selected_matrix_function: BdGMatrixFunction,
    integration: IntegrationMethod,
    keys: list[tuple[int, ...]],
    density_selection: DensitySelection,
    prepared_node,
    workspace_dtype: np.dtype,
) -> tuple[_tb_type, _tb_type]:
    if isinstance(selected_matrix_function, DirectDiagonalization):
        density = density_block(
            selected_matrix_function,
            shift_by_mu(matrix, root.mu, q_diag, dtype=workspace_dtype),
            np.eye(q_diag.size, dtype=workspace_dtype),
            kT=kT,
            q_diag=q_diag,
            derivative=False,
            tolerance=integration.density_matrix_tol,
            workspace_dtype=workspace_dtype,
        ).block
        return density_selection.values_and_errors_to_tb(
            density_selection.values_from_assembled_matrix(density),
            np.zeros(density_selection.value_count, dtype=float),
        )

    assert prepared_node is not None
    density = prepared_node.density_values_from_charge_order(root.mu)
    return density_selection.values_and_errors_to_tb(
        density,
        np.zeros(density_selection.value_count, dtype=float),
    )


def _bdg_zero_dim_info(root, *, filling_tol: float, integration: IntegrationMethod):
    return FixedFillingInfo(
        mu=root.mu,
        charge=root.charge,
        charge_error=root.charge_error,
        dcharge_dmu=float("nan") if root.derivative is None else root.derivative,
        charge_evaluations=root.charge_evaluations,
        charge_integration_calls=root.charge_evaluations,
        density_integration_calls=1,
        charge_n_kernel_evals=1,
        density_n_kernel_evals=1,
        n_kernel_evals=1,
        unique_evals=1,
        charge_n_evaluator_evals=root.charge_evaluations,
        density_n_evaluator_evals=1,
        n_evaluator_evals=root.charge_evaluations + 1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
        charge_integral_atol=filling_tol,
        density_atol=integration.density_matrix_tol,
        density_rtol=0.0,
        error_estimate_available=True,
    )


def _solve_bdg_zero_dim(
    *,
    hamiltonian: _tb_type,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
    q_diag: np.ndarray,
    selected_matrix_function: BdGMatrixFunction,
    filling_indices,
    filling_weights: np.ndarray,
    density_selection: DensitySelection,
) -> DensityMatrixResult:
    from meanfi.density.filling import mu_bracket_for_bdg

    workspace_dtype = workspace_complex_dtype(integration)
    matrix = hamiltonian[tuple()]
    filling_block = basis_block(q_diag.size, filling_indices, dtype=workspace_dtype)
    prepared_node = _bdg_zero_dim_prepared_node(
        matrix=matrix,
        kT=kT,
        q_diag=q_diag,
        selected_matrix_function=selected_matrix_function,
        filling_tol=filling_tol,
        density_tolerance=integration.density_matrix_tol,
        filling_indices=filling_indices,
        filling_weights=filling_weights,
        density_selection=density_selection,
        workspace_dtype=workspace_dtype,
    )
    evaluate_charge = _bdg_zero_dim_charge_evaluator(
        matrix=matrix,
        kT=kT,
        q_diag=q_diag,
        selected_matrix_function=selected_matrix_function,
        filling_indices=filling_indices,
        filling_weights=filling_weights,
        filling_block=filling_block,
        integration=integration,
        prepared_node=prepared_node,
        workspace_dtype=workspace_dtype,
    )

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=lambda: mu_bracket_for_bdg(hamiltonian, kT),
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        use_derivative=(
            kT > 0
            and not (
                prepared_node is not None
                and selected_matrix_function.rational_scheme == "aaa"
            )
        ),
    )

    density_matrix, density_matrix_error = _bdg_zero_dim_density(
        matrix=matrix,
        kT=kT,
        root=root,
        q_diag=q_diag,
        selected_matrix_function=selected_matrix_function,
        integration=integration,
        keys=keys,
        density_selection=density_selection,
        prepared_node=prepared_node,
        workspace_dtype=workspace_dtype,
    )
    raw_info = _bdg_zero_dim_info(
        root, filling_tol=filling_tol, integration=integration
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
                charge_evaluations=raw_info.charge_evaluations,
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


@dataclass(frozen=True)
class BdGFixedFillingContext:
    model: object
    hamiltonian: _tb_type
    keys: list[tuple[int, ...]]
    integration: IntegrationMethod
    filling_tol: float
    mu_tol: float
    max_charge_evaluations: int | None
    mu_guess: float
    density_selection: DensitySelection | None
    q_diag: np.ndarray
    selected_matrix_function: BdGMatrixFunction
    filling_indices: tuple[int, ...]
    filling_weights: np.ndarray


def build_bdg_problem(
    model,
    meanfield: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    density_selection: DensitySelection | None = None,
) -> BdGFixedFillingContext:
    """Normalize a BdG fixed-filling problem before executor selection."""

    validate_integration_method(integration, kT=model.kT)
    hamiltonian = model.bdg_hamiltonian_from_meanfield(meanfield)
    selected_matrix_function = resolve_bdg_matrix_function(
        integration,
        hamiltonian,
        kT=model.kT,
    )
    q_diag = charge_diagonal(model._ndof)
    filling_indices = tuple(range(model._ndof))
    filling_weights = np.ones(model._ndof, dtype=float)
    return BdGFixedFillingContext(
        model=model,
        hamiltonian=hamiltonian,
        keys=keys,
        integration=integration,
        filling_tol=effective_bdg_filling_tol(
            filling_tol=filling_tol,
            density_matrix_tol=getattr(integration, "density_matrix_tol"),
            filling_weights=filling_weights,
        ),
        mu_tol=0.0,
        max_charge_evaluations=None,
        mu_guess=0.0,
        density_selection=density_selection,
        q_diag=q_diag,
        selected_matrix_function=selected_matrix_function,
        filling_indices=filling_indices,
        filling_weights=filling_weights,
    )


def _solve_bdg_uniform_grid_fixed_filling(
    context: BdGFixedFillingContext,
) -> DensityMatrixResult:
    integration = context.integration
    assert isinstance(integration, UniformGrid)
    model = context.model
    workspace_dtype = workspace_complex_dtype(integration)
    resolved_density_selection = (
        context.density_selection
        if context.density_selection is not None
        else full_density_selection(context.keys, size=2 * model._ndof)
    )
    bundle = build_uniform_grid_node_bundle(
        context.hamiltonian,
        kT=model.kT,
        nk=integration.nk,
        keys=context.keys,
        matrix_function=context.selected_matrix_function,
        q_diag=context.q_diag,
        trace_weights_diag=np.concatenate(
            [np.ones(model._ndof, dtype=float), np.zeros(model._ndof, dtype=float)]
        ),
        charge_tolerance=context.filling_tol,
        density_tolerance=integration.density_matrix_tol,
        density_selection=resolved_density_selection,
        workspace_dtype=workspace_dtype,
    )
    density_matrix, mu, resolved_filling, info = _uniform_fixed_filling_from_nodes(
        bundle,
        hamiltonian=context.hamiltonian,
        integration=integration,
        filling=model.filling,
        mu_guess=context.mu_guess,
        filling_tol=context.filling_tol,
        mu_tol=context.mu_tol,
        max_charge_evaluations=context.max_charge_evaluations,
        mu_bracket_builder=lambda: mu_bracket_for_bdg(context.hamiltonian, model.kT),
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
            hamiltonian=context.hamiltonian,
            n_kernel_evals=info.n_kernel_evals,
            n_evaluator_evals=info.n_evaluator_evals,
            charge_evaluations=info.charge_evaluations,
            charge_integration_calls=info.charge_integration_calls,
            density_integration_calls=info.density_integration_calls,
            error_estimate_available=False,
        ),
        keys=context.keys,
    )


def _solve_bdg_adaptive_quadrature_fixed_filling(
    context: BdGFixedFillingContext,
) -> DensityMatrixResult:
    integration = context.integration
    assert isinstance(integration, AdaptiveQuadrature)
    backend = build_bdg_backend(
        context.hamiltonian,
        keys=context.keys,
        kT=context.model.kT,
        q_diag=context.q_diag,
        matrix_function=context.selected_matrix_function,
        filling_indices=context.filling_indices,
        filling_weights=context.filling_weights,
        tolerance=integration.density_matrix_tol,
        charge_tolerance=context.filling_tol,
        density_selection=context.density_selection,
        workspace_dtype=workspace_complex_dtype(integration),
    )
    density_matrix, density_matrix_error, raw_info = _solve_quadrature_fixed_filling(
        backend,
        filling=context.model.filling,
        mu_guess=context.mu_guess,
        rule=integration.rule,
        batch_size=integration.batch_size,
        filling_tol=context.filling_tol,
        mu_tol=context.mu_tol,
        max_charge_evaluations=context.max_charge_evaluations,
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
        target_filling=context.model.filling,
        integration=integration,
        keys=context.keys,
    )


def solve_bdg_density_fixed_filling(
    model,
    meanfield: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
    density_selection: DensitySelection | None = None,
) -> DensityMatrixResult:
    if mu_tol <= 0:
        raise ValueError("mu_tol must be positive")
    if max_charge_evaluations is not None and max_charge_evaluations <= 0:
        raise ValueError("max_charge_evaluations must be positive")

    context = build_bdg_problem(
        model,
        meanfield,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        density_selection=density_selection,
    )
    context = BdGFixedFillingContext(
        **{
            **context.__dict__,
            "mu_tol": mu_tol,
            "max_charge_evaluations": max_charge_evaluations,
            "mu_guess": mu_guess,
        }
    )

    if model._ndim == 0:
        density_selection = context.density_selection
        if density_selection is None:
            density_selection = full_density_selection(
                context.keys,
                size=2 * model._ndof,
            )
        return _solve_bdg_zero_dim(
            hamiltonian=context.hamiltonian,
            filling=model.filling,
            kT=model.kT,
            keys=context.keys,
            integration=context.integration,
            filling_tol=context.filling_tol,
            mu_tol=context.mu_tol,
            max_charge_evaluations=context.max_charge_evaluations,
            mu_guess=context.mu_guess,
            q_diag=context.q_diag,
            selected_matrix_function=context.selected_matrix_function,
            filling_indices=context.filling_indices,
            filling_weights=context.filling_weights,
            density_selection=density_selection,
        )

    if isinstance(context.integration, UniformGrid):
        return _solve_bdg_uniform_grid_fixed_filling(context)
    if isinstance(context.integration, AdaptiveQuadrature):
        return _solve_bdg_adaptive_quadrature_fixed_filling(context)
    raise NotImplementedError(
        "BdG density currently requires UniformGrid at kT == 0 and "
        "AdaptiveQuadrature or UniformGrid at kT > 0"
    )
