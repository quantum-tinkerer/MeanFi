from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warnings

from meanfi.core.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.core.validation import tb_dimension
from meanfi.integrate.matrix_functions import (
    BdGMatrixFunction,
    density_block,
    matrix_function_label,
    shift_by_mu,
)
from meanfi.superconducting.bdg import mu_bracket_for_bdg
from meanfi.tb.tb import _tb_type

from ..matrix_functions import basis_block
from .normal_backend import integration_bounds, quadrature_prefactor, split_density_result
from .runtime import QuadratureBackend


def _tb_k_matrix(hamiltonian: _tb_type, point: np.ndarray):
    accumulator = None
    for key, matrix in hamiltonian.items():
        phase = np.exp(-1j * np.dot(point, np.asarray(key, dtype=float)))
        term = matrix * phase
        accumulator = term if accumulator is None else accumulator + term
    if accumulator is None:
        raise ValueError("BdG Hamiltonian cannot be empty")
    return accumulator


def _payload_helpers(hamiltonian: _tb_type):
    from meanfi.core.matrix import as_sparse, is_sparse_like, matrix_shape, sparse_module

    first_matrix = next(iter(hamiltonian.values()))
    shape = matrix_shape(first_matrix)
    if not any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
        size = shape[0]

        def kernel(points: np.ndarray) -> np.ndarray:
            payload = np.empty((points.shape[0], size * size), dtype=complex)
            for index, point in enumerate(points):
                payload[index] = np.asarray(
                    _tb_k_matrix(hamiltonian, point),
                    dtype=complex,
                ).reshape(-1)
            return payload

        def matrix_from_payload(payload_row: np.ndarray):
            return np.asarray(payload_row, dtype=complex).reshape(shape)

        return kernel, matrix_from_payload

    sparse = sparse_module()
    structural = sparse.csr_matrix(shape, dtype=np.int8)
    term_payloads: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []
    for key, matrix in hamiltonian.items():
        csr_matrix = as_sparse(matrix).tocsr()
        coo_matrix = csr_matrix.tocoo()
        if coo_matrix.nnz > 0:
            structural = structural + sparse.csr_matrix(
                (
                    np.ones(coo_matrix.nnz, dtype=np.int8),
                    (coo_matrix.row, coo_matrix.col),
                ),
                shape=shape,
            )
        term_payloads.append(
            (
                key,
                np.stack([coo_matrix.row, coo_matrix.col], axis=1).astype(int, copy=False),
                np.asarray(coo_matrix.data, dtype=complex),
            )
        )

    structural.sum_duplicates()
    structural = structural.tocsr()
    structural.sort_indices()
    structural.data[:] = 1

    locations: dict[tuple[int, int], int] = {}
    for row in range(shape[0]):
        start = int(structural.indptr[row])
        end = int(structural.indptr[row + 1])
        for offset in range(start, end):
            locations[(row, int(structural.indices[offset]))] = offset

    sparse_terms: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []
    for key, coordinates, data in term_payloads:
        if coordinates.size == 0:
            sparse_terms.append((key, np.empty(0, dtype=int), data))
            continue
        positions = np.fromiter(
            (locations[(int(row), int(col))] for row, col in coordinates),
            dtype=int,
            count=coordinates.shape[0],
        )
        sparse_terms.append((key, positions, data))

    def kernel(points: np.ndarray) -> np.ndarray:
        payload = np.zeros((points.shape[0], structural.nnz), dtype=complex)
        for key, positions, data in sparse_terms:
            if positions.size == 0:
                continue
            phase = np.exp(-1j * np.dot(points, np.asarray(key, dtype=float)))
            payload[:, positions] += phase[:, np.newaxis] * data[np.newaxis, :]
        return payload

    def matrix_from_payload(payload_row: np.ndarray):
        return sparse.csr_matrix(
            (
                np.asarray(payload_row, dtype=complex),
                structural.indices,
                structural.indptr,
            ),
            shape=shape,
        )

    return kernel, matrix_from_payload


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
):
    prefactor = quadrature_prefactor(ndim)
    size = q_diag.size
    filling_block = basis_block(size, filling_indices)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        values = np.empty((points.shape[0], 2), dtype=float)
        for index, point in enumerate(points):
            matrix = shift_by_mu(matrix_from_payload(payload[index]), mu, q_diag)
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
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(keys, dtype=float)
    size = q_diag.size
    density_basis = np.eye(size, dtype=complex)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        values = np.empty((points.shape[0], size * size * len(keys)), dtype=complex)
        for index, point in enumerate(points):
            matrix = shift_by_mu(matrix_from_payload(payload[index]), mu, q_diag)
            result = density_block(
                matrix_function,
                matrix,
                density_basis,
                kT=kT,
                q_diag=q_diag,
                derivative=False,
                tolerance=tolerance,
            )
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = (
                result.block[..., np.newaxis] * phase[np.newaxis, np.newaxis, :]
            ).reshape(-1)
        return prefactor * values

    return evaluator


def _split_charge(estimate: np.ndarray, error: np.ndarray) -> tuple[float, float, float]:
    estimate = np.asarray(estimate)
    error = np.asarray(error)
    return float(np.real(estimate[0])), float(abs(error[0])), float(np.real(estimate[1]))


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
) -> QuadratureBackend:
    ndim = tb_dimension(hamiltonian)
    kernel, matrix_from_payload = _payload_helpers(hamiltonian)

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

    return QuadratureBackend(
        bounds=integration_bounds(ndim),
        kernel=kernel,
        charge_evaluator=_charge_evaluator(
            ndim=ndim,
            kT=kT,
            q_diag=q_diag,
            matrix_function=matrix_function,
            tolerance=tolerance,
            filling_indices=filling_indices,
            filling_weights=filling_weights,
            matrix_from_payload=matrix_from_payload,
        ),
        density_evaluator=_density_evaluator(
            ndim=ndim,
            kT=kT,
            q_diag=q_diag,
            matrix_function=matrix_function,
            tolerance=tolerance,
            keys=keys,
            matrix_from_payload=matrix_from_payload,
        ),
        split_charge_result=_split_charge,
        split_density_result=lambda estimate, error: split_density_result(
            estimate,
            error,
            q_diag.size,
            keys,
        ),
        density_info_builder=density_info_builder,
        fixed_filling_info_builder=fixed_filling_info_builder,
        mu_bracket=lambda: mu_bracket_for_bdg(hamiltonian, kT),
    )
