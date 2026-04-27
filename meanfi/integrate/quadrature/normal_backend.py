from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.core.filling import mu_bracket
from meanfi.core.results import DensityIntegrationInfo, FixedFillingInfo
from meanfi.core.validation import tb_dimension, tb_orbital_count
from meanfi.integrate.matrix_functions import (
    ChebyshevFOE,
    DirectDiagonalization,
    RationalFOE,
    resolve_matrix_function,
)
from meanfi.integrate.matrix_functions.prepared_normal import (
    PreparedNormalChebyshevNode,
    PreparedNormalRationalNode,
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


def prepared_payload_builder(
    *,
    matrix_function: ChebyshevFOE | RationalFOE,
    matrix_from_payload,
    kT: float,
    tolerance: float,
):
    def builder(points: np.ndarray, payload: np.ndarray) -> list[Any]:
        del points
        prepared = []
        for payload_row in payload:
            matrix = matrix_from_payload(payload_row)
            if isinstance(matrix_function, ChebyshevFOE):
                prepared.append(
                    PreparedNormalChebyshevNode(
                        matrix,
                        kT=kT,
                        options=matrix_function,
                        tolerance=tolerance,
                    )
                )
            elif isinstance(matrix_function, RationalFOE):
                prepared.append(
                    PreparedNormalRationalNode(
                        matrix,
                        kT=kT,
                        options=matrix_function,
                        tolerance=tolerance,
                    )
                )
            else:  # pragma: no cover - guarded by build_normal_backend
                raise TypeError("Unsupported prepared normal matrix function")
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


def prepared_density_evaluator(ndim: int, keys: list[tuple[int, ...]]):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(keys, dtype=float)

    def evaluator(points: np.ndarray, payload: list[Any], mu: float) -> np.ndarray:
        values = np.empty(
            (points.shape[0], payload[0].size * payload[0].size * len(keys)),
            dtype=complex,
        )
        for index, (point, prepared) in enumerate(zip(points, payload, strict=True)):
            density_k = prepared.density(mu)
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = (
                density_k[..., np.newaxis] * phase[np.newaxis, np.newaxis, :]
            ).reshape(-1)
        return prefactor * values

    return evaluator


def split_charge_result(
    estimate: np.ndarray, error: np.ndarray
) -> tuple[float, float, float]:
    """Return charge, charge error, and dcharge/dmu from a quadrature result."""

    estimate = np.asarray(estimate)
    error = np.asarray(error)
    return (
        float(np.real(estimate[0])),
        float(np.abs(error[0])),
        float(np.real(estimate[1])),
    )


def split_density_result(
    estimate: np.ndarray,
    error: np.ndarray,
    ndof: int,
    keys: list[tuple[int, ...]],
) -> tuple[_tb_type, _tb_type]:
    """Convert vectorized density estimates to tight-binding dictionaries."""

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
) -> QuadratureBackend:
    ndim = tb_dimension(hamiltonian)
    ndof = tb_orbital_count(hamiltonian)
    matrix_function = resolve_matrix_function(getattr(integration, "matrix_function", None))

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

    if isinstance(matrix_function, DirectDiagonalization):
        kernel = spectral_payload(hamiltonian)
        payload_builder = None
        charge_kernel = charge_evaluator(ndim, ndof, kT)
        density_kernel = density_evaluator(ndim, ndof, keys, kT)
    elif isinstance(matrix_function, (ChebyshevFOE, RationalFOE)):
        kernel, matrix_from_payload = build_tb_payload_helpers(hamiltonian)
        payload_builder = prepared_payload_builder(
            matrix_function=matrix_function,
            matrix_from_payload=matrix_from_payload,
            kT=kT,
            tolerance=integration.density_matrix_tol,
        )
        charge_kernel = prepared_charge_evaluator(ndim)
        density_kernel = prepared_density_evaluator(ndim, keys)
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
        ),
        density_info_builder=integration_stats,
        fixed_filling_info_builder=fixed_filling_info_builder,
        mu_bracket=lambda: mu_bracket(hamiltonian, kT),
    )
