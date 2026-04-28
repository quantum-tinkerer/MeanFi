from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from meanfi.core.matrix import as_sparse, is_sparse_like, sparse_linalg_module

from .base import RationalFOE, _BlockResult
from .common import (
    _derivative_convergence,
    scalar_derivative_converged,
    spectral_interval,
    trace_probe_block,
    weighted_trace_from_basis_result,
    workspace_matrix,
)
from ..occupations import fermi_dirac
from .common import shift_by_mu


@lru_cache(maxsize=None)
def _ozaki_exact_poles_and_residues(pole_count: int) -> tuple[np.ndarray, np.ndarray]:
    if pole_count <= 0:
        raise ValueError("pole_count must be positive")

    matrix_size = 2 * int(pole_count)
    diagonal_weights = np.arange(1, 2 * matrix_size, 2, dtype=float)
    off_diagonal = -0.5 / np.sqrt(diagonal_weights[:-1] * diagonal_weights[1:])
    continued_fraction = np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(continued_fraction)
    negative = eigenvalues < 0.0
    if int(np.count_nonzero(negative)) != int(pole_count):
        raise ValueError("Ozaki pole construction returned an unexpected pole count")

    selected_eigenvalues = eigenvalues[negative]
    selected_vectors = eigenvectors[0, negative]
    poles = -1.0 / selected_eigenvalues
    residues = -0.25 * np.square(selected_vectors) * np.square(poles)
    return np.asarray(poles, dtype=float), np.asarray(residues, dtype=float)


def _irls_minimax_fit(
    design: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 6,
) -> np.ndarray:
    weights = np.ones(target.size, dtype=float)
    coeffs = np.zeros(design.shape[1], dtype=float)
    for _ in range(iterations):
        weighted_design = weights[:, np.newaxis] * design
        weighted_target = weights * target
        coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        residual = target - design @ coeffs
        weights = 1.0 / np.maximum(np.abs(residual), 1e-8)
    return coeffs


def _minimax_terms(
    pole_count: int,
    *,
    max_poles: int,
    lower: float,
    upper: float,
    kT: float,
) -> tuple[complex, np.ndarray, np.ndarray]:
    poles, _ = _ozaki_exact_poles_and_residues(max(int(max_poles), int(pole_count)))
    shifts = 1j * np.asarray(poles[: int(pole_count)], dtype=float) * float(kT)
    n_samples = max(256, 16 * int(max_poles))
    x = np.cos(np.pi * (np.arange(n_samples, dtype=float) + 0.5) / n_samples)
    center = 0.5 * (upper + lower)
    scale = 0.5 * max(upper - lower, 1e-12)
    energies = center + scale * x
    target = fermi_dirac(energies, kT, 0.0)

    design = np.empty((n_samples, 1 + 2 * shifts.size), dtype=float)
    design[:, 0] = 1.0
    for index, shift in enumerate(shifts):
        basis = 1.0 / (energies - shift)
        design[:, 1 + 2 * index] = 2.0 * np.real(basis)
        design[:, 2 + 2 * index] = -2.0 * np.imag(basis)
    coeffs = _irls_minimax_fit(design, np.asarray(target, dtype=float))
    constant = complex(coeffs[0])
    residues = coeffs[1::2].astype(float) + 1j * coeffs[2::2].astype(float)
    return constant, np.asarray(shifts, dtype=complex), np.asarray(residues, dtype=complex)


def _scheme_terms(
    options: RationalFOE,
    pole_count: int,
    *,
    lower: float,
    upper: float,
    kT: float,
) -> tuple[complex, np.ndarray, np.ndarray]:
    if options.rational_scheme == "ozaki":
        poles, residues = _ozaki_exact_poles_and_residues(int(pole_count))
        return (
            complex(0.5),
            1j * np.asarray(poles, dtype=float) * float(kT),
            np.asarray(residues, dtype=float) * float(kT),
        )
    if options.rational_scheme == "minimax":
        return _minimax_terms(
            pole_count,
            max_poles=int(options.max_poles),
            lower=lower,
            upper=upper,
            kT=kT,
        )
    raise ValueError(f"Unsupported RationalFOE scheme: {options.rational_scheme}")


def _dense_shifted_matrix(matrix: np.ndarray, shift: complex) -> np.ndarray:
    shifted = np.array(matrix, copy=True)
    diagonal = shifted.diagonal().copy()
    diagonal -= complex(shift)
    np.fill_diagonal(shifted, diagonal)
    return shifted


def _sparse_shifted_lu(matrix: Any, shift: complex):
    shifted = as_sparse(matrix).tocsc()
    shifted = shifted.copy()
    diagonal = np.asarray(shifted.diagonal(), dtype=complex)
    diagonal -= complex(shift)
    shifted.setdiag(diagonal)
    return sparse_linalg_module().splu(shifted)


def _evaluate_rational_terms(
    matrix: Any,
    block: np.ndarray,
    *,
    constant: complex,
    shifts: np.ndarray,
    residues: np.ndarray,
    q_diag: np.ndarray,
    derivative: bool,
    workspace_dtype: np.dtype,
    lu_cache: dict[complex, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[complex, Any] | None]:
    matrix = workspace_matrix(matrix, workspace_dtype)
    block = np.asarray(block, dtype=workspace_dtype)
    density_result = np.asarray(constant * block, dtype=complex)
    derivative_block = np.zeros_like(block, dtype=complex) if derivative else None

    if is_sparse_like(matrix):
        active_cache = {} if lu_cache is None else dict(lu_cache)
        for shift, residue in zip(shifts, residues, strict=True):
            key = complex(shift)
            lu = active_cache.get(key)
            if lu is None:
                lu = _sparse_shifted_lu(matrix, key)
                active_cache[key] = lu
            rhs = np.asarray(block, dtype=workspace_dtype)
            y = np.asarray(lu.solve(rhs), dtype=complex)
            y_adj = np.asarray(
                lu.solve(np.asarray(block, dtype=workspace_dtype), trans="H"),
                dtype=complex,
            )
            density_result = density_result + residue * y + np.conjugate(residue) * y_adj

            if derivative:
                rhs = np.asarray(q_diag[:, np.newaxis] * y, dtype=workspace_dtype)
                rhs_adj = np.asarray(q_diag[:, np.newaxis] * y_adj, dtype=workspace_dtype)
                z = np.asarray(lu.solve(rhs), dtype=complex)
                z_adj = np.asarray(lu.solve(rhs_adj, trans="H"), dtype=complex)
                derivative_block = (
                    derivative_block
                    + residue * z
                    + np.conjugate(residue) * z_adj
                )
        return density_result, derivative_block, active_cache

    dense_matrix = np.asarray(matrix, dtype=workspace_dtype)
    for shift, residue in zip(shifts, residues, strict=True):
        shifted = _dense_shifted_matrix(dense_matrix, shift)
        shifted_adjoint = shifted.conj().T
        y = np.linalg.solve(shifted, block)
        y_adj = np.linalg.solve(shifted_adjoint, block)
        density_result = density_result + residue * y + np.conjugate(residue) * y_adj

        if derivative:
            rhs = q_diag[:, np.newaxis] * y
            rhs_adj = q_diag[:, np.newaxis] * y_adj
            z = np.linalg.solve(shifted, rhs)
            z_adj = np.linalg.solve(shifted_adjoint, rhs_adj)
            derivative_block = derivative_block + residue * z + np.conjugate(residue) * z_adj

    return density_result, derivative_block, lu_cache


def _evaluate_rational_poles(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    pole_count: int,
    derivative: bool,
    options: RationalFOE,
    workspace_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray | None]:
    lower, upper = spectral_interval(matrix)
    constant, shifts, residues = _scheme_terms(
        options,
        pole_count,
        lower=lower,
        upper=upper,
        kT=kT,
    )
    density_result, derivative_block, _ = _evaluate_rational_terms(
        matrix,
        block,
        constant=constant,
        shifts=shifts,
        residues=residues,
        q_diag=q_diag,
        derivative=derivative,
        workspace_dtype=workspace_dtype,
    )
    return density_result, derivative_block


def _rational_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    options: RationalFOE,
    derivative_trace_monitor=None,
    derivative_context: str | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> _BlockResult:
    if options.rational_scheme == "minimax":
        full_block, full_derivative = _evaluate_rational_poles(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            pole_count=int(options.max_poles),
            derivative=derivative,
            options=options,
            workspace_dtype=workspace_dtype,
        )
        return _BlockResult(
            block=full_block,
            derivative_block=full_derivative,
            error=0.0,
            order=int(options.max_poles),
        )

    accepted_block = None
    accepted_derivative = None
    accepted_error = float("inf")
    accepted_order = None
    derivative_converged = True

    half_poles = int(options.initial_poles)
    half_block, half_derivative = _evaluate_rational_poles(
        matrix,
        block,
        kT=kT,
        q_diag=q_diag,
        pole_count=half_poles,
        derivative=derivative,
        options=options,
        workspace_dtype=workspace_dtype,
    )

    while 2 * half_poles <= int(options.max_poles):
        pole_count = 2 * half_poles
        full_block, full_derivative = _evaluate_rational_poles(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            pole_count=pole_count,
            derivative=derivative,
            options=options,
            workspace_dtype=workspace_dtype,
        )
        accepted_error = float(np.max(np.abs(full_block - half_block)))

        if derivative:
            derivative_converged, _derivative_error = _derivative_convergence(
                full_derivative,
                half_derivative,
                derivative=derivative,
                dn_dmu_rtol=options.dn_dmu_rtol,
                derivative_trace_monitor=derivative_trace_monitor,
                derivative_context=derivative_context,
                matrix_function_name="Rational FOE",
            )

        accepted_block = full_block
        accepted_derivative = full_derivative
        accepted_order = pole_count
        if accepted_error <= tolerance and derivative_converged:
            break

        half_poles = pole_count
        half_block = full_block
        half_derivative = full_derivative

    if accepted_error > tolerance or (derivative and not derivative_converged):
        raise ValueError("Rational FOE did not converge within max_poles")

    return _BlockResult(
        block=accepted_block,
        derivative_block=accepted_derivative,
        error=accepted_error,
        order=accepted_order,
    )


class PreparedRationalNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        q_diag: np.ndarray,
        options: RationalFOE,
        charge_tolerance: float,
        workspace_dtype: np.dtype = np.dtype(complex),
        trace_weights_diag: np.ndarray | None = None,
    ) -> None:
        self.workspace_dtype = np.dtype(workspace_dtype)
        self.matrix = workspace_matrix(matrix, self.workspace_dtype)
        self.kT = float(kT)
        self.q_diag = np.asarray(q_diag, dtype=float)
        self.options = options
        self.charge_tolerance = float(charge_tolerance)
        self.size = int(getattr(matrix, "shape")[0])
        self._trace_weights = (
            np.ones(self.size, dtype=float)
            if trace_weights_diag is None
            else np.asarray(trace_weights_diag, dtype=float)
        )
        self._trace_estimator = options.trace_estimator
        self._trace_columns = (
            np.flatnonzero(np.abs(self._trace_weights) > 0.0).astype(int, copy=False)
            if self._trace_estimator == "exact"
            else None
        )
        if self._trace_estimator == "exact":
            if self._trace_columns is None or self._trace_columns.size == 0:
                raise ValueError("Exact weighted trace requires at least one nonzero trace weight")
            self._trace_basis = np.zeros(
                (self.size, self._trace_columns.size),
                dtype=self.workspace_dtype,
            )
            self._trace_basis[
                self._trace_columns,
                np.arange(self._trace_columns.size),
            ] = 1.0
        else:
            self._trace_basis = trace_probe_block(
                self.size,
                estimator=self._trace_estimator,
                trace_probes=options.trace_probes,
                trace_seed=options.trace_seed,
                dtype=self.workspace_dtype,
            )
        self._charge_cache: dict[float, tuple[float, float, int]] = {}
        self._last_mu: float | None = None
        self._last_lu_cache: dict[complex, Any] = {}

    def _trace_scalar(self, block: np.ndarray) -> float:
        return weighted_trace_from_basis_result(
            block,
            self._trace_basis,
            weights_diag=self._trace_weights,
            estimator=self._trace_estimator,
            trace_columns=self._trace_columns,
        )

    def _evaluate_terms_for_basis(
        self,
        mu: float,
        basis: np.ndarray,
        *,
        pole_count: int,
        derivative: bool,
        lu_cache: dict[complex, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[complex, Any] | None]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        lower, upper = spectral_interval(shifted)
        constant, shifts, residues = _scheme_terms(
            self.options,
            pole_count,
            lower=lower,
            upper=upper,
            kT=self.kT,
        )
        return _evaluate_rational_terms(
            shifted,
            np.asarray(basis, dtype=self.workspace_dtype),
            constant=constant,
            shifts=shifts,
            residues=residues,
            q_diag=self.q_diag,
            derivative=derivative,
            workspace_dtype=self.workspace_dtype,
            lu_cache=lu_cache,
        )

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, derivative, _order = self._charge_cache[mu]
            return charge, derivative

        if self.options.rational_scheme == "minimax":
            full_block, full_derivative_block, lu_cache = self._evaluate_terms_for_basis(
                mu,
                self._trace_basis,
                pole_count=int(self.options.max_poles),
                derivative=True,
                lu_cache=None,
            )
            full_charge = self._trace_scalar(full_block)
            full_derivative = (
                0.0 if full_derivative_block is None else self._trace_scalar(full_derivative_block)
            )
            self._charge_cache[mu] = (
                full_charge,
                full_derivative,
                int(self.options.max_poles),
            )
            self._last_mu = float(mu)
            self._last_lu_cache = {} if lu_cache is None else dict(lu_cache)
            return full_charge, full_derivative

        half_poles = int(self.options.initial_poles)
        half_block, half_derivative_block, lu_cache = self._evaluate_terms_for_basis(
            mu,
            self._trace_basis,
            pole_count=half_poles,
            derivative=True,
            lu_cache=None,
        )
        half_charge = self._trace_scalar(half_block)
        half_derivative = (
            0.0 if half_derivative_block is None else self._trace_scalar(half_derivative_block)
        )

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            full_block, full_derivative_block, lu_cache = self._evaluate_terms_for_basis(
                mu,
                self._trace_basis,
                pole_count=pole_count,
                derivative=True,
                lu_cache=lu_cache if self.options.rational_scheme == "ozaki" else None,
            )
            full_charge = self._trace_scalar(full_block)
            full_derivative = (
                0.0 if full_derivative_block is None else self._trace_scalar(full_derivative_block)
            )

            if (
                abs(full_charge - half_charge) <= self.charge_tolerance
                and scalar_derivative_converged(
                    full_derivative,
                    half_derivative,
                    rtol=self.options.dn_dmu_rtol,
                )
            ):
                self._charge_cache[mu] = (full_charge, full_derivative, pole_count)
                self._last_mu = float(mu)
                self._last_lu_cache = {} if lu_cache is None else dict(lu_cache)
                return full_charge, full_derivative

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_charge = full_charge
            half_derivative = full_derivative

    def density_columns_from_charge_order(self, mu: float, basis: np.ndarray) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError("Charge pole count is unavailable for the requested chemical potential")
        lu_cache = self._last_lu_cache if self._last_mu == float(mu) else None
        block, _derivative_block, lu_cache = self._evaluate_terms_for_basis(
            mu,
            basis,
            pole_count=charge_cache[2],
            derivative=False,
            lu_cache=lu_cache,
        )
        if lu_cache is not None:
            self._last_mu = float(mu)
            self._last_lu_cache = dict(lu_cache)
        return block

    def density_from_charge_order(self, mu: float) -> np.ndarray:
        block = self.density_columns_from_charge_order(
            mu,
            np.eye(self.size, dtype=self.workspace_dtype),
        )
        return 0.5 * (block + block.conj().T)
