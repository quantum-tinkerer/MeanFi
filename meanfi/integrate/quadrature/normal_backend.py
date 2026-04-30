from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.core.filling import mu_bracket
from meanfi.core.matrix import is_sparse_like
from meanfi.core.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.core.validation import tb_dimension, tb_orbital_count
from meanfi.integrate.density_support import (
    DensityEntrySupport,
    full_density_entry_support,
    workspace_complex_dtype,
)
from meanfi.integrate.matrix_functions import (
    ChebyshevFOE,
    DirectDiagonalization,
    RationalFOE,
    basis_block,
    density_block,
    resolve_matrix_function,
    shift_by_mu,
)
from meanfi.integrate.matrix_functions.prepared_normal import PreparedNormalChebyshevNode
from meanfi.integrate.matrix_functions.rational import (
    PreparedMumpsRationalNode,
    PreparedRationalNode,
)
from meanfi.integrate.methods import AdaptiveQuadrature
from meanfi.integrate.occupations import fermi_dirac
from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import tb_to_kfunc

from .payloads import build_tb_payload_helpers
from .runtime import QuadratureBackend


def quadrature_prefactor(ndim: int) -> float:
    """Return the Brillouin-zone integration prefactor."""

    return 1.0 if ndim == 0 else 1.0 / (2.0 * np.pi) ** ndim


def integration_bounds(ndim: int) -> tuple[list[float], list[float]]:
    """Return the Brillouin-zone bounds for stateful quadrature."""

    return ([-np.pi] * ndim, [np.pi] * ndim)


def integration_stats(result) -> DensityIntegrationInfo:
    """Convert integrator metadata to the internal info dataclass."""

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


def resolve_normal_matrix_function(
    selected: object | None,
) -> DirectDiagonalization | ChebyshevFOE | RationalFOE:
    return resolve_matrix_function(selected)


def spectral_payload(hamiltonian: _tb_type):
    """Build the cached eigensystem payload consumed by stateful quadrature."""

    hkfunc = tb_to_kfunc(hamiltonian)
    ndof = tb_orbital_count(hamiltonian)

    def kernel(points: np.ndarray) -> np.ndarray:
        h_k = hkfunc(points)
        eigenvalues, eigenvectors = np.linalg.eigh(h_k)
        return np.concatenate(
            [eigenvalues, eigenvectors.reshape(points.shape[0], ndof * ndof)],
            axis=-1,
        )

    return kernel


def charge_evaluator(ndim: int, ndof: int, kT: float):
    """Return an evaluator for charge and charge-derivative integrands."""

    prefactor = quadrature_prefactor(ndim)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        del points
        eigenvalues = payload[:, :ndof].real
        occupation = fermi_dirac(eigenvalues, kT, mu)
        dcharge_dmu = occupation * (1.0 - occupation) / kT
        charge = np.sum(occupation, axis=-1, keepdims=True)
        derivative = np.sum(dcharge_dmu, axis=-1, keepdims=True)
        return prefactor * np.concatenate([charge, derivative], axis=-1)

    return evaluator


def density_evaluator(ndim: int, ndof: int, keys: list[tuple[int, ...]], kT: float):
    """Return an evaluator for density-matrix Fourier components."""

    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(keys, dtype=float)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        eigenvalues = payload[:, :ndof].real
        eigenvectors = payload[:, ndof:].reshape(points.shape[0], ndof, ndof)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density_k = (
            eigenvectors
            * occupation[:, np.newaxis, :]
            @ eigenvectors.conj().transpose(0, 2, 1)
        )
        phase = np.exp(1j * np.dot(points, keys_array.T))
        density_terms = (
            density_k[..., np.newaxis] * phase[:, np.newaxis, np.newaxis, :]
        ).reshape(points.shape[0], -1)
        return prefactor * density_terms

    return evaluator


def selected_density_evaluator(
    ndim: int,
    ndof: int,
    kT: float,
    density_support: DensityEntrySupport,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(density_support.keys, dtype=float)
    selected_columns = density_support.selected_columns

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        eigenvalues = payload[:, :ndof].real
        eigenvectors = payload[:, ndof:].reshape(points.shape[0], ndof, ndof)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        selected_rows = np.take(eigenvectors.conj(), selected_columns, axis=1)
        density_columns = eigenvectors @ np.swapaxes(
            occupation[:, np.newaxis, :] * selected_rows,
            -1,
            -2,
        )
        phase = np.exp(1j * np.dot(points, keys_array.T))
        values = np.empty((points.shape[0], density_support.output_size), dtype=complex)
        for index in range(points.shape[0]):
            values[index] = density_support.pack_columns(
                density_columns[index],
                phases=phase[index],
            )
        return prefactor * values

    return evaluator


def prepared_payload_builder(
    *,
    matrix_function: ChebyshevFOE,
    matrix_from_payload,
    kT: float,
    charge_tolerance: float | None,
    workspace_dtype: np.dtype,
    trace_weights_diag: np.ndarray | None = None,
):
    def builder(points: np.ndarray, payload: np.ndarray) -> list[Any]:
        del points
        prepared = []
        for payload_row in payload:
            matrix = matrix_from_payload(payload_row)
            prepared.append(
                PreparedNormalChebyshevNode(
                    matrix,
                    kT=kT,
                    options=matrix_function,
                    charge_tolerance=charge_tolerance,
                    workspace_dtype=workspace_dtype,
                    trace_weights_diag=trace_weights_diag,
                )
            )
        return prepared

    return builder


def prepared_rational_payload_builder(
    *,
    matrix_function: RationalFOE,
    matrix_from_payload,
    kT: float,
    charge_tolerance: float,
    workspace_dtype: np.dtype,
    q_diag: np.ndarray,
    trace_weights_diag: np.ndarray | None = None,
):
    def builder(points: np.ndarray, payload: np.ndarray) -> list[Any]:
        del points
        prepared = []
        for payload_row in payload:
            matrix = matrix_from_payload(payload_row)
            prepared.append(
                PreparedRationalNode(
                    matrix,
                    kT=kT,
                    q_diag=q_diag,
                    options=matrix_function,
                    charge_tolerance=charge_tolerance,
                    workspace_dtype=workspace_dtype,
                    trace_weights_diag=trace_weights_diag,
                )
            )
        return prepared

    return builder


def prepared_mumps_rational_payload_builder(
    *,
    matrix_function: RationalFOE,
    matrix_from_payload,
    kT: float,
    charge_tolerance: float,
    density_tolerance: float,
    density_support: DensityEntrySupport,
    workspace_dtype: np.dtype,
    q_diag: np.ndarray,
    trace_weights_diag: np.ndarray,
):
    shared_aaa_interval_cache = []

    def builder(points: np.ndarray, payload: np.ndarray) -> list[Any]:
        del points
        prepared = []
        for payload_row in payload:
            matrix = matrix_from_payload(payload_row)
            prepared.append(
                PreparedMumpsRationalNode(
                    matrix,
                    kT=kT,
                    q_diag=q_diag,
                    options=matrix_function,
                    charge_tolerance=charge_tolerance,
                    density_support=density_support,
                    density_tolerance=density_tolerance,
                    workspace_dtype=workspace_dtype,
                    trace_weights_diag=trace_weights_diag,
                    shared_aaa_interval_cache=shared_aaa_interval_cache,
                )
            )
        return prepared

    return builder


def prepared_charge_evaluator(ndim: int):
    prefactor = quadrature_prefactor(ndim)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        del points
        values = np.empty((len(payload), 2), dtype=float)
        for index, prepared in enumerate(payload):
            charge, derivative = prepared.charge_and_derivative(mu)
            values[index, 0] = charge
            values[index, 1] = derivative
        return prefactor * values

    return evaluator


def prepared_charge_only_evaluator(ndim: int):
    prefactor = quadrature_prefactor(ndim)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        del points
        values = np.empty((len(payload), 1), dtype=float)
        for index, prepared in enumerate(payload):
            charge, _derivative = prepared.charge_and_derivative(mu)
            values[index, 0] = charge
        return prefactor * values

    return evaluator


def prepared_transient_density_evaluator(
    ndim: int,
    keys: list[tuple[int, ...]],
    *,
    density_tolerance: float,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(keys, dtype=float)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        values = np.empty(
            (points.shape[0], payload[0].size * payload[0].size * len(keys)),
            dtype=complex,
        )
        for index, (point, prepared) in enumerate(zip(points, payload, strict=True)):
            density_k = prepared.density(mu, tolerance=density_tolerance)
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = (
                density_k[..., np.newaxis] * phase[np.newaxis, np.newaxis, :]
            ).reshape(-1)
        return prefactor * values

    return evaluator


def prepared_frozen_density_evaluator(ndim: int, keys: list[tuple[int, ...]]):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(keys, dtype=float)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        values = np.empty(
            (points.shape[0], payload[0].size * payload[0].size * len(keys)),
            dtype=complex,
        )
        for index, (point, prepared) in enumerate(zip(points, payload, strict=True)):
            density_k = prepared.density_from_charge_order(mu)
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = (
                density_k[..., np.newaxis] * phase[np.newaxis, np.newaxis, :]
            ).reshape(-1)
        return prefactor * values

    return evaluator


def prepared_selected_frozen_density_evaluator(
    ndim: int,
    density_support: DensityEntrySupport,
    *,
    workspace_dtype: np.dtype,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(density_support.keys, dtype=float)
    basis = density_support.basis_block(dtype=workspace_dtype)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        values = np.empty((points.shape[0], density_support.output_size), dtype=complex)
        for index, (point, prepared) in enumerate(zip(points, payload, strict=True)):
            density_columns = prepared.density_columns_from_charge_order(mu, basis)
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = density_support.pack_columns(
                density_columns,
                phases=phase,
            )
        return prefactor * values

    return evaluator


def prepared_selected_transient_density_evaluator(
    ndim: int,
    density_support: DensityEntrySupport,
    *,
    tolerance: float,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(density_support.keys, dtype=float)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        values = np.empty((points.shape[0], density_support.output_size), dtype=complex)
        for index, (point, prepared) in enumerate(zip(points, payload, strict=True)):
            density_columns = prepared.density_columns(mu, tolerance=tolerance)
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = density_support.pack_columns(
                density_columns,
                phases=phase,
            )
        return prefactor * values

    return evaluator


def _local_trace(block: np.ndarray) -> float:
    return float(np.real(np.trace(block)))


def rational_charge_evaluator(
    ndim: int,
    ndof: int,
    *,
    kT: float,
    matrix_function: RationalFOE,
    tolerance: float,
    matrix_from_payload,
    workspace_dtype: np.dtype,
):
    prefactor = quadrature_prefactor(ndim)
    q_diag = np.ones(ndof, dtype=float)
    trace_basis = basis_block(ndof, range(ndof), dtype=workspace_dtype)

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
                trace_basis,
                kT=kT,
                q_diag=q_diag,
                derivative=True,
                tolerance=tolerance,
                workspace_dtype=workspace_dtype,
            )
            values[index, 0] = _local_trace(result.block)
            derivative_block = result.derivative_block
            values[index, 1] = 0.0 if derivative_block is None else _local_trace(derivative_block)
        return prefactor * values

    return evaluator


def rational_density_evaluator(
    ndim: int,
    ndof: int,
    *,
    kT: float,
    matrix_function: RationalFOE,
    tolerance: float,
    keys: list[tuple[int, ...]],
    matrix_from_payload,
    density_support: DensityEntrySupport | None,
    workspace_dtype: np.dtype,
):
    prefactor = quadrature_prefactor(ndim)
    q_diag = np.ones(ndof, dtype=float)
    keys_array = np.asarray(
        density_support.keys if density_support is not None else keys,
        dtype=float,
    )
    density_basis = (
        density_support.basis_block(dtype=workspace_dtype)
        if density_support is not None
        else np.eye(ndof, dtype=workspace_dtype)
    )

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        output_size = (
            density_support.output_size
            if density_support is not None
            else ndof * ndof * len(keys)
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


def split_charge_result(
    estimate: np.ndarray, error: np.ndarray
) -> tuple[float, float, float | None]:
    """Return charge, charge error, and dcharge/dmu from a quadrature result."""

    estimate = np.asarray(estimate)
    error = np.asarray(error)
    derivative = (
        float(np.real(estimate[1]))
        if estimate.size > 1
        else None
    )
    return (
        float(np.real(estimate[0])),
        float(np.abs(error[0])),
        derivative,
    )


def split_density_result(
    estimate: np.ndarray,
    error: np.ndarray,
    ndof: int,
    keys: list[tuple[int, ...]],
    density_support: DensityEntrySupport | None = None,
) -> tuple[_tb_type, _tb_type]:
    """Convert vectorized density estimates to tight-binding dictionaries."""

    if density_support is not None:
        return density_support.expand_entries(estimate, error)

    estimate = np.asarray(estimate).reshape(ndof, ndof, len(keys))
    error = np.asarray(error).reshape(ndof, ndof, len(keys))

    rho = {}
    rho_error = {}
    for index, key in enumerate(keys):
        rho[key] = estimate[..., index]
        rho_error[key] = error[..., index]
    return rho, rho_error


def build_normal_backend(
    hamiltonian: _tb_type,
    *,
    integration: AdaptiveQuadrature,
    keys: list[tuple[int, ...]],
    kT: float,
    fixed_filling_tolerance: float | None = None,
    density_entry_support: DensityEntrySupport | None = None,
) -> QuadratureBackend:
    ndim = tb_dimension(hamiltonian)
    ndof = tb_orbital_count(hamiltonian)
    workspace_dtype = workspace_complex_dtype(integration)
    matrix_function = resolve_normal_matrix_function(
        getattr(integration, "matrix_function", None)
    )
    density_support = density_entry_support
    use_sparse_mumps = isinstance(matrix_function, RationalFOE) and any(
        is_sparse_like(matrix) for matrix in hamiltonian.values()
    )
    mumps_density_support = (
        full_density_entry_support(keys, size=ndof)
        if use_sparse_mumps and density_support is None
        else density_support
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
        density_stats = integration_stats(density_result)
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
            n_cached_nodes=density_stats.n_cached_nodes,
            n_leaves=density_stats.n_leaves,
            n_leaf_nodes=density_stats.n_leaf_nodes,
            subdivisions=charge_subdivisions + density_stats.subdivisions,
            charge_integral_atol=charge_integral_atol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            error_estimate_available=True,
        )

    freeze_density_mesh = False
    charge_has_derivative = True
    if isinstance(matrix_function, DirectDiagonalization):
        kernel = spectral_payload(hamiltonian)
        payload_builder = None
        charge_kernel = charge_evaluator(ndim, ndof, kT)
        density_kernel = density_evaluator(ndim, ndof, keys, kT)
    elif isinstance(matrix_function, ChebyshevFOE):
        kernel, matrix_from_payload = build_tb_payload_helpers(hamiltonian)
        payload_builder = prepared_payload_builder(
            matrix_function=matrix_function,
            matrix_from_payload=matrix_from_payload,
            kT=kT,
            charge_tolerance=fixed_filling_tolerance,
            workspace_dtype=workspace_dtype,
            trace_weights_diag=np.ones(ndof, dtype=float),
        )
        charge_kernel = prepared_charge_evaluator(ndim)
        if fixed_filling_tolerance is None:
            density_kernel = prepared_transient_density_evaluator(
                ndim,
                keys,
                density_tolerance=integration.density_matrix_tol,
            )
        else:
            density_kernel = (
                prepared_selected_frozen_density_evaluator(
                    ndim,
                    density_support,
                    workspace_dtype=workspace_dtype,
                )
                if density_support is not None
                else prepared_frozen_density_evaluator(ndim, keys)
            )
            freeze_density_mesh = True
    elif isinstance(matrix_function, RationalFOE):
        kernel, matrix_from_payload = build_tb_payload_helpers(hamiltonian)
        if use_sparse_mumps:
            if mumps_density_support is None:  # pragma: no cover - guarded above
                raise ValueError("Sparse MUMPS-backed RationalFOE requires a density support descriptor")
            payload_builder = prepared_mumps_rational_payload_builder(
                matrix_function=matrix_function,
                matrix_from_payload=matrix_from_payload,
                kT=kT,
                charge_tolerance=(
                    integration.density_matrix_tol
                    if fixed_filling_tolerance is None
                    else fixed_filling_tolerance
                ),
                density_tolerance=integration.density_matrix_tol,
                density_support=mumps_density_support,
                workspace_dtype=workspace_dtype,
                q_diag=np.ones(ndof, dtype=float),
                trace_weights_diag=np.ones(ndof, dtype=float),
            )
            if fixed_filling_tolerance is None:
                charge_kernel = prepared_charge_evaluator(ndim)
                density_kernel = prepared_selected_transient_density_evaluator(
                    ndim,
                    mumps_density_support,
                    tolerance=integration.density_matrix_tol,
                )
            else:
                charge_kernel = prepared_charge_only_evaluator(ndim)
                density_kernel = prepared_selected_frozen_density_evaluator(
                    ndim,
                    mumps_density_support,
                    workspace_dtype=workspace_dtype,
                )
                freeze_density_mesh = True
                charge_has_derivative = False
        elif fixed_filling_tolerance is None:
            payload_builder = None
            charge_kernel = rational_charge_evaluator(
                ndim,
                ndof,
                kT=kT,
                matrix_function=matrix_function,
                tolerance=integration.density_matrix_tol,
                matrix_from_payload=matrix_from_payload,
                workspace_dtype=workspace_dtype,
            )
            density_kernel = rational_density_evaluator(
                ndim,
                ndof,
                kT=kT,
                matrix_function=matrix_function,
                tolerance=integration.density_matrix_tol,
                keys=keys,
                matrix_from_payload=matrix_from_payload,
                density_support=density_support,
                workspace_dtype=workspace_dtype,
            )
        else:
            payload_builder = prepared_rational_payload_builder(
                matrix_function=matrix_function,
                matrix_from_payload=matrix_from_payload,
                kT=kT,
                charge_tolerance=fixed_filling_tolerance,
                workspace_dtype=workspace_dtype,
                q_diag=np.ones(ndof, dtype=float),
                trace_weights_diag=np.ones(ndof, dtype=float),
            )
            charge_kernel = prepared_charge_evaluator(ndim)
            density_kernel = (
                prepared_selected_frozen_density_evaluator(
                    ndim,
                    density_support,
                    workspace_dtype=workspace_dtype,
                )
                if density_support is not None
                else prepared_frozen_density_evaluator(ndim, keys)
            )
            freeze_density_mesh = True
    else:  # pragma: no cover - guarded by resolve_matrix_function
        raise TypeError("AdaptiveQuadrature.matrix_function must be a BdGMatrixFunction")

    return QuadratureBackend(
        bounds=integration_bounds(ndim),
        kernel=kernel,
        payload_builder=payload_builder,
        charge_evaluator=charge_kernel,
        density_evaluator=density_kernel,
        split_charge_result=split_charge_result,
        split_density_result=lambda estimate, error: split_density_result(
            estimate,
            error,
            ndof,
            keys,
            None
            if isinstance(matrix_function, DirectDiagonalization)
            else (mumps_density_support if use_sparse_mumps else density_support),
        ),
        density_info_builder=integration_stats,
        fixed_filling_info_builder=fixed_filling_info_builder,
        mu_bracket=lambda: mu_bracket(hamiltonian, kT),
        freeze_density_mesh=freeze_density_mesh,
        charge_has_derivative=charge_has_derivative,
    )
