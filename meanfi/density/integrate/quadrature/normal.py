from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.density.filling import mu_bracket
from meanfi.tb.ops import is_sparse_like
from meanfi.results import DensityIntegrationInfo
from meanfi.tb.validate import tb_dimension, tb_orbital_count
from meanfi.density.integrate.workspace import workspace_complex_dtype
from meanfi.space.density_selection import DensitySelection
from meanfi.space.density_selection import full_density_selection
from meanfi.density.kpoint.matrix_functions import (
    DirectDiagonalization,
    RationalFOE,
    resolve_sparse_default_matrix_function,
    selected_density_values_from_eigensystem,
)
from meanfi.density.kpoint.matrix_functions.rational import (
    PreparedMumpsRationalNode,
)
from meanfi.density.integrate.methods import AdaptiveQuadrature
from meanfi.density.kpoint.occupations import fermi_dirac
from meanfi.tb.ops import _tb_type
from meanfi.tb.transforms import tb_to_kfunc

from .payloads import build_tb_payload_helpers
from .runtime import QuadratureBackend, density_integration_info, fixed_filling_info


def quadrature_prefactor(ndim: int) -> float:
    """Return the Brillouin-zone integration prefactor."""

    return 1.0 if ndim == 0 else 1.0 / (2.0 * np.pi) ** ndim


def integration_bounds(ndim: int) -> tuple[list[float], list[float]]:
    """Return the Brillouin-zone bounds for stateful quadrature."""

    return ([-np.pi] * ndim, [np.pi] * ndim)


def integration_stats(result) -> DensityIntegrationInfo:
    """Convert integrator metadata to the internal info dataclass."""

    return density_integration_info(result)


def resolve_normal_matrix_function(
    selected: object | None,
    hamiltonian: _tb_type,
) -> DirectDiagonalization | RationalFOE:
    return resolve_sparse_default_matrix_function(
        selected,
        hamiltonian,
        parameter_name="AdaptiveQuadrature.matrix_function",
    )


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
    density_selection: DensitySelection,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(density_selection.keys, dtype=float)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        eigenvalues = payload[:, :ndof].real
        eigenvectors = payload[:, ndof:].reshape(points.shape[0], ndof, ndof)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        phase = np.exp(1j * np.dot(points, keys_array.T))
        values = selected_density_values_from_eigensystem(
            eigenvectors,
            occupation,
            density_selection,
            phases=phase,
        )
        return prefactor * values

    return evaluator


def prepared_mumps_rational_payload_builder(
    *,
    matrix_function: RationalFOE,
    matrix_from_payload,
    kT: float,
    charge_tolerance: float,
    density_tolerance: float,
    density_selection: DensitySelection,
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
                    density_selection=density_selection,
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
    density_selection: DensitySelection,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(density_selection.keys, dtype=float)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        values = np.empty(
            (points.shape[0], density_selection.value_count), dtype=complex
        )
        for index, (point, prepared) in enumerate(zip(points, payload, strict=True)):
            density_values = prepared.density_values_from_charge_order(mu)
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = density_selection.phase_values(density_values, phase)
        return prefactor * values

    return evaluator


def prepared_selected_transient_density_evaluator(
    ndim: int,
    density_selection: DensitySelection,
    *,
    tolerance: float,
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(density_selection.keys, dtype=float)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        values = np.empty(
            (points.shape[0], density_selection.value_count), dtype=complex
        )
        for index, (point, prepared) in enumerate(zip(points, payload, strict=True)):
            density_values = prepared.density_values(mu, tolerance=tolerance)
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = density_selection.phase_values(density_values, phase)
        return prefactor * values

    return evaluator


def split_charge_result(
    estimate: np.ndarray, error: np.ndarray
) -> tuple[float, float, float | None]:
    """Return charge, charge error, and dcharge/dmu from a quadrature result."""

    estimate = np.asarray(estimate)
    error = np.asarray(error)
    derivative = float(np.real(estimate[1])) if estimate.size > 1 else None
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
    density_selection: DensitySelection | None = None,
) -> tuple[_tb_type, _tb_type]:
    """Convert vectorized density estimates to tight-binding dictionaries."""

    if density_selection is not None:
        return density_selection.values_and_errors_to_tb(estimate, error)

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
    density_selection: DensitySelection | None = None,
) -> QuadratureBackend:
    ndim = tb_dimension(hamiltonian)
    ndof = tb_orbital_count(hamiltonian)
    workspace_dtype = workspace_complex_dtype(integration)
    matrix_function = resolve_normal_matrix_function(
        getattr(integration, "matrix_function", None),
        hamiltonian,
    )
    density_selection = density_selection
    use_sparse_mumps = isinstance(matrix_function, RationalFOE) and any(
        is_sparse_like(matrix) for matrix in hamiltonian.values()
    )
    mumps_density_selection = (
        full_density_selection(keys, size=ndof)
        if use_sparse_mumps and density_selection is None
        else density_selection
    )

    freeze_density_mesh = False
    charge_has_derivative = True
    if isinstance(matrix_function, DirectDiagonalization):
        kernel = spectral_payload(hamiltonian)
        payload_builder = None
        charge_kernel = charge_evaluator(ndim, ndof, kT)
        density_kernel = (
            selected_density_evaluator(ndim, ndof, kT, density_selection)
            if density_selection is not None
            else density_evaluator(ndim, ndof, keys, kT)
        )
    elif isinstance(matrix_function, RationalFOE):
        kernel, matrix_from_payload = build_tb_payload_helpers(hamiltonian)
        if not use_sparse_mumps:
            raise ValueError(
                "AdaptiveQuadrature RationalFOE is supported only for sparse matrices"
            )
        if mumps_density_selection is None:  # pragma: no cover - guarded above
            raise ValueError(
                "Sparse MUMPS-backed RationalFOE requires a density selection"
            )
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
            density_selection=mumps_density_selection,
            workspace_dtype=workspace_dtype,
            q_diag=np.ones(ndof, dtype=float),
            trace_weights_diag=np.ones(ndof, dtype=float),
        )
        if fixed_filling_tolerance is None:
            charge_kernel = prepared_charge_evaluator(ndim)
            density_kernel = prepared_selected_transient_density_evaluator(
                ndim,
                mumps_density_selection,
                tolerance=integration.density_matrix_tol,
            )
        else:
            charge_kernel = prepared_charge_only_evaluator(ndim)
            density_kernel = prepared_selected_frozen_density_evaluator(
                ndim,
                mumps_density_selection,
            )
            freeze_density_mesh = True
            charge_has_derivative = False
    else:  # pragma: no cover - guarded by resolve_normal_matrix_function
        raise TypeError(
            "AdaptiveQuadrature.matrix_function must be DirectDiagonalization or RationalFOE"
        )

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
            (
                density_selection
                if isinstance(matrix_function, DirectDiagonalization)
                else (
                    mumps_density_selection if use_sparse_mumps else density_selection
                )
            ),
        ),
        density_info_builder=integration_stats,
        fixed_filling_info_builder=fixed_filling_info,
        mu_bracket=lambda: mu_bracket(hamiltonian, kT),
        freeze_density_mesh=freeze_density_mesh,
        charge_has_derivative=charge_has_derivative,
    )
