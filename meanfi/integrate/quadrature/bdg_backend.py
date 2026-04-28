from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warnings

from meanfi.core.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.core.validation import tb_dimension
from meanfi.integrate.density_support import DensityEntrySupport
from meanfi.integrate.matrix_functions import (
    BdGMatrixFunction,
    ChebyshevFOE,
    DirectDiagonalization,
    RationalFOE,
    density_block,
    matrix_function_label,
    shift_by_mu,
)
from meanfi.integrate.matrix_functions.prepared_normal import PreparedShiftedChebyshevNode
from meanfi.integrate.matrix_functions.rational import PreparedRationalNode
from meanfi.superconducting.bdg import mu_bracket_for_bdg
from meanfi.tb.tb import _tb_type

from ..matrix_functions import basis_block
from .normal_backend import (
    integration_bounds,
    prepared_charge_evaluator,
    prepared_frozen_density_evaluator,
    prepared_selected_frozen_density_evaluator,
    quadrature_prefactor,
    split_density_result,
)
from .payloads import build_tb_payload_helpers
from .runtime import QuadratureBackend


def _local_filling(block: np.ndarray, indices: Sequence[int], weights: np.ndarray) -> float:
    values = block[np.asarray(indices, dtype=int), np.arange(len(indices))]
    return float(np.real(np.sum(weights * values)))


def _charge_evaluator(
    *,
    ndim: int,
    kT: float,
    q_diag: np.ndarray,
    matrix_function: BdGMatrixFunction,
    tolerance: float,
    filling_indices: Sequence[int],
    filling_weights: np.ndarray,
    matrix_from_payload,
    workspace_dtype: np.dtype,
):
    prefactor = quadrature_prefactor(ndim)
    size = q_diag.size
    filling_block = basis_block(size, filling_indices, dtype=workspace_dtype)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        values = np.empty((points.shape[0], 2), dtype=float)
        for index, point in enumerate(points):
            matrix = shift_by_mu(
                matrix_from_payload(payload[index]),
                mu,
                q_diag,
                dtype=workspace_dtype,
            )
            result = density_block(
                matrix_function,
                matrix,
                filling_block,
                kT=kT,
                q_diag=q_diag,
                derivative=True,
                tolerance=tolerance,
                derivative_trace_monitor=lambda block: _local_filling(
                    block,
                    filling_indices,
                    filling_weights,
                ),
                derivative_context=f"k={tuple(np.asarray(point, dtype=float))}, mu={float(mu):.12g}",
                workspace_dtype=workspace_dtype,
            )
            derivative_block = result.derivative_block
            derivative = (
                0.0
                if derivative_block is None
                else _local_filling(derivative_block, filling_indices, filling_weights)
            )
            if derivative < 0.0:
                warnings.warn(
                    "BdG local dn/dmu was negative after "
                    + matrix_function_label(matrix_function)
                    + " convergence"
                    + f" (k={tuple(np.asarray(point, dtype=float))}, mu={float(mu):.12g})",
                    RuntimeWarning,
                )
                derivative = 0.0
            values[index, 0] = _local_filling(
                result.block,
                filling_indices,
                filling_weights,
            )
            values[index, 1] = derivative
        return prefactor * values

    return evaluator


def _density_evaluator(
    *,
    ndim: int,
    kT: float,
    q_diag: np.ndarray,
    matrix_function: BdGMatrixFunction,
    tolerance: float,
    keys: list[tuple[int, ...]],
    matrix_from_payload,
    density_support: DensityEntrySupport | None,
    workspace_dtype: np.dtype,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(
        density_support.keys if density_support is not None else keys,
        dtype=float,
    )
    size = q_diag.size
    density_basis = (
        density_support.basis_block(dtype=workspace_dtype)
        if density_support is not None
        else np.eye(size, dtype=workspace_dtype)
    )

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        output_size = (
            density_support.output_size
            if density_support is not None
            else size * size * len(keys)
        )
        values = np.empty((points.shape[0], output_size), dtype=complex)
        for index, point in enumerate(points):
            matrix = shift_by_mu(
                matrix_from_payload(payload[index]),
                mu,
                q_diag,
                dtype=workspace_dtype,
            )
            result = density_block(
                matrix_function,
                matrix,
                density_basis,
                kT=kT,
                q_diag=q_diag,
                derivative=False,
                tolerance=tolerance,
                workspace_dtype=workspace_dtype,
            )
            phase = np.exp(1j * np.dot(point, keys_array.T))
            if density_support is not None:
                values[index] = density_support.pack_columns(
                    result.block,
                    phases=phase,
                )
            else:
                values[index] = (
                    result.block[..., np.newaxis] * phase[np.newaxis, np.newaxis, :]
                ).reshape(-1)
        return prefactor * values

    return evaluator


def _split_charge(estimate: np.ndarray, error: np.ndarray) -> tuple[float, float, float]:
    estimate = np.asarray(estimate)
    error = np.asarray(error)
    return float(np.real(estimate[0])), float(abs(error[0])), float(np.real(estimate[1]))


def _prepared_payload_builder(
    *,
    matrix_function: BdGMatrixFunction,
    kT: float,
    q_diag: np.ndarray,
    filling_weights: np.ndarray,
    charge_tolerance: float,
    matrix_from_payload,
    workspace_dtype: np.dtype,
):
    def builder(points: np.ndarray, payload: np.ndarray) -> list[object]:
        del points
        prepared: list[object] = []
        for payload_row in payload:
            matrix = matrix_from_payload(payload_row)
            if isinstance(matrix_function, ChebyshevFOE):
                prepared.append(
                    PreparedShiftedChebyshevNode(
                        matrix,
                        kT=kT,
                        q_diag=q_diag,
                        options=matrix_function,
                        charge_tolerance=charge_tolerance,
                        workspace_dtype=workspace_dtype,
                        trace_weights_diag=filling_weights,
                    )
                )
            elif isinstance(matrix_function, RationalFOE):
                prepared.append(
                    PreparedRationalNode(
                        matrix,
                        kT=kT,
                        q_diag=q_diag,
                        options=matrix_function,
                        charge_tolerance=charge_tolerance,
                        workspace_dtype=workspace_dtype,
                        trace_weights_diag=filling_weights,
                    )
                )
            else:  # pragma: no cover - guarded by caller
                raise TypeError("Prepared payloads require a matrix-function backend")
        return prepared

    return builder


def build_bdg_backend(
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    kT: float,
    q_diag: np.ndarray,
    matrix_function: BdGMatrixFunction,
    filling_indices: Sequence[int],
    filling_weights: np.ndarray,
    tolerance: float,
    charge_tolerance: float,
    density_entry_support: DensityEntrySupport | None,
    workspace_dtype: np.dtype,
) -> QuadratureBackend:
    ndim = tb_dimension(hamiltonian)
    kernel, matrix_from_payload = build_tb_payload_helpers(hamiltonian)

    def density_info_builder(result) -> DensityIntegrationInfo:
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

    def fixed_filling_info_builder(
        *,
        mu: float,
        charge: float,
        charge_error: float,
        derivative: float,
        root_iterations: int,
        charge_integration_calls: int,
        charge_kernel_evals: int,
        charge_evaluator_evals: int,
        charge_subdivisions: int,
        density_result,
        charge_integral_atol: float,
        density_atol: float,
        density_rtol: float,
    ) -> FixedFillingInfo:
        return FixedFillingInfo(
            mu=mu,
            charge=charge,
            charge_error=charge_error,
            dcharge_dmu=derivative,
            root_iterations=root_iterations,
            charge_integration_calls=charge_integration_calls,
            density_integration_calls=1,
            charge_n_kernel_evals=charge_kernel_evals,
            density_n_kernel_evals=int(density_result.n_kernel_evals),
            n_kernel_evals=charge_kernel_evals + int(density_result.n_kernel_evals),
            unique_evals=charge_kernel_evals + int(density_result.n_kernel_evals),
            charge_n_evaluator_evals=charge_evaluator_evals,
            density_n_evaluator_evals=int(density_result.n_evaluator_evals),
            n_evaluator_evals=charge_evaluator_evals + int(density_result.n_evaluator_evals),
            n_cached_nodes=int(
                getattr(density_result, "n_cached_nodes", getattr(density_result, "n_leaf_nodes", 0))
            ),
            n_leaves=int(getattr(density_result, "n_leaves", 0)),
            n_leaf_nodes=int(getattr(density_result, "n_leaf_nodes", 0)),
            subdivisions=charge_subdivisions + int(getattr(density_result, "subdivisions", 0)),
            charge_integral_atol=charge_integral_atol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            error_estimate_available=True,
        )

    payload_builder = None
    charge_evaluator = _charge_evaluator(
        ndim=ndim,
        kT=kT,
        q_diag=q_diag,
        matrix_function=matrix_function,
        tolerance=tolerance,
        filling_indices=filling_indices,
        filling_weights=filling_weights,
        matrix_from_payload=matrix_from_payload,
        workspace_dtype=workspace_dtype,
    )
    density_evaluator = _density_evaluator(
        ndim=ndim,
        kT=kT,
        q_diag=q_diag,
        matrix_function=matrix_function,
        tolerance=tolerance,
        keys=keys,
        matrix_from_payload=matrix_from_payload,
        density_support=(
            None if isinstance(matrix_function, DirectDiagonalization) else density_entry_support
        ),
        workspace_dtype=workspace_dtype,
    )
    freeze_density_mesh = False
    if not isinstance(matrix_function, DirectDiagonalization):
        trace_weights = np.zeros(q_diag.size, dtype=float)
        trace_weights[np.asarray(filling_indices, dtype=int)] = np.asarray(
            filling_weights,
            dtype=float,
        )
        payload_builder = _prepared_payload_builder(
            matrix_function=matrix_function,
            kT=kT,
            q_diag=q_diag,
            filling_weights=trace_weights,
            charge_tolerance=charge_tolerance,
            matrix_from_payload=matrix_from_payload,
            workspace_dtype=workspace_dtype,
        )
        charge_evaluator = prepared_charge_evaluator(ndim)
        density_evaluator = (
            prepared_selected_frozen_density_evaluator(
                ndim,
                density_entry_support,
                workspace_dtype=workspace_dtype,
            )
            if density_entry_support is not None
            else prepared_frozen_density_evaluator(ndim, keys)
        )
        freeze_density_mesh = True

    return QuadratureBackend(
        bounds=integration_bounds(ndim),
        kernel=kernel,
        payload_builder=payload_builder,
        charge_evaluator=charge_evaluator,
        density_evaluator=density_evaluator,
        split_charge_result=_split_charge,
        split_density_result=(
            (lambda estimate, error: density_entry_support.expand_entries(estimate, error))
            if density_entry_support is not None
            and not isinstance(matrix_function, DirectDiagonalization)
            else lambda estimate, error: split_density_result(
                estimate,
                error,
                q_diag.size,
                keys,
            )
        ),
        density_info_builder=density_info_builder,
        fixed_filling_info_builder=fixed_filling_info_builder,
        mu_bracket=lambda: mu_bracket_for_bdg(hamiltonian, kT),
        freeze_density_mesh=freeze_density_mesh,
    )
