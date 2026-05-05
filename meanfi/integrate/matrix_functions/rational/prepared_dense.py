from __future__ import annotations

from typing import Any

import numpy as np

from ..base import RationalFOE
from ..common import (
    scalar_derivative_converged,
    spectral_interval,
    workspace_matrix,
    shift_by_mu,
)
from .common import _evaluate_rational_terms
from .scheme import _aaa_terms_for_interval, _scheme_terms


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
        self._trace_columns = np.flatnonzero(np.abs(self._trace_weights) > 0.0).astype(
            int,
            copy=False,
        )
        if self._trace_columns.size == 0:
            raise ValueError(
                "Exact weighted trace requires at least one nonzero trace weight"
            )
        self._trace_basis = np.zeros(
            (self.size, self._trace_columns.size),
            dtype=self.workspace_dtype,
        )
        self._trace_basis[
            self._trace_columns,
            np.arange(self._trace_columns.size),
        ] = 1.0
        self._charge_cache: dict[float, tuple[float, float, int]] = {}
        self._last_mu: float | None = None
        self._last_lu_cache: dict[complex, Any] = {}
        self._last_constant: complex = complex(0.0)
        self._last_shifts = np.empty(0, dtype=np.complex128)
        self._last_residues = np.empty(0, dtype=np.complex128)

    def _trace_scalar(self, block: np.ndarray) -> float:
        values = block[self._trace_columns, np.arange(self._trace_columns.size)]
        return float(np.real(np.sum(self._trace_weights[self._trace_columns] * values)))

    def _charge_scalar_tolerance(self) -> float:
        weight_sum = float(np.sum(np.abs(self._trace_weights[self._trace_columns])))
        if weight_sum <= 0.0:
            return self.charge_tolerance
        return max(
            np.finfo(float).eps,
            float(self.charge_tolerance) / weight_sum,
        )

    def _evaluate_known_terms_for_basis(
        self,
        mu: float,
        basis: np.ndarray,
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
        derivative: bool,
        lu_cache: dict[complex, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[complex, Any] | None]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        return _evaluate_rational_terms(
            shifted,
            np.asarray(basis, dtype=self.workspace_dtype),
            constant=constant,
            shifts=np.asarray(shifts, dtype=np.complex128),
            residues=np.asarray(residues, dtype=np.complex128),
            q_diag=self.q_diag,
            derivative=derivative,
            workspace_dtype=self.workspace_dtype,
            lu_cache=lu_cache,
        )

    def _certified_aaa_terms(
        self, mu: float
    ) -> tuple[complex, np.ndarray, np.ndarray, int]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        lower, upper = spectral_interval(shifted)
        terms, _builder = _aaa_terms_for_interval(
            pole_cap=int(self.options.max_poles),
            lower=lower,
            upper=upper,
            kT=self.kT,
            initial_poles=int(self.options.initial_poles),
            scalar_tolerance=self._charge_scalar_tolerance(),
        )
        return (
            terms.constant,
            np.asarray(terms.shifts, dtype=np.complex128),
            np.asarray(terms.residues, dtype=np.complex128),
            int(terms.pole_count),
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
        if self.options.rational_scheme == "aaa":
            constant, shifts, residues, _pole_count = self._certified_aaa_terms(mu)
            return self._evaluate_known_terms_for_basis(
                mu,
                basis,
                constant=constant,
                shifts=shifts,
                residues=residues,
                derivative=derivative,
                lu_cache=lu_cache,
            )
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
        return self._evaluate_known_terms_for_basis(
            mu,
            basis,
            constant=constant,
            shifts=shifts,
            residues=residues,
            derivative=derivative,
            lu_cache=lu_cache,
        )

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, derivative, _order = self._charge_cache[mu]
            return charge, derivative

        if self.options.rational_scheme == "aaa":
            constant, shifts, residues, pole_count = self._certified_aaa_terms(mu)
            block, _derivative_block, lu_cache = self._evaluate_known_terms_for_basis(
                mu,
                self._trace_basis,
                constant=constant,
                shifts=shifts,
                residues=residues,
                derivative=False,
                lu_cache=None,
            )
            charge = self._trace_scalar(block)
            derivative = float("nan")
            self._charge_cache[mu] = (charge, derivative, pole_count)
            self._last_mu = float(mu)
            self._last_lu_cache = {} if lu_cache is None else dict(lu_cache)
            self._last_constant = constant
            self._last_shifts = np.asarray(shifts, dtype=np.complex128)
            self._last_residues = np.asarray(residues, dtype=np.complex128)
            return charge, derivative

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
            0.0
            if half_derivative_block is None
            else self._trace_scalar(half_derivative_block)
        )

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            full_block, full_derivative_block, lu_cache = (
                self._evaluate_terms_for_basis(
                    mu,
                    self._trace_basis,
                    pole_count=pole_count,
                    derivative=True,
                    lu_cache=lu_cache,
                )
            )
            full_charge = self._trace_scalar(full_block)
            full_derivative = (
                0.0
                if full_derivative_block is None
                else self._trace_scalar(full_derivative_block)
            )

            if abs(
                full_charge - half_charge
            ) <= self.charge_tolerance and scalar_derivative_converged(
                full_derivative,
                half_derivative,
                rtol=self.options.dn_dmu_rtol,
            ):
                self._charge_cache[mu] = (full_charge, full_derivative, pole_count)
                self._last_mu = float(mu)
                self._last_lu_cache = {} if lu_cache is None else dict(lu_cache)
                self._last_constant = complex(0.0)
                self._last_shifts = np.empty(0, dtype=np.complex128)
                self._last_residues = np.empty(0, dtype=np.complex128)
                return full_charge, full_derivative

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_charge = full_charge
            half_derivative = full_derivative

    def density_columns_from_charge_order(
        self, mu: float, basis: np.ndarray
    ) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError(
                "Charge pole count is unavailable for the requested chemical potential"
            )
        if self.options.rational_scheme == "aaa":
            if self._last_mu != float(mu):
                raise ValueError(
                    "Charge pole data is unavailable for the requested chemical potential"
                )
            block, _derivative_block, lu_cache = self._evaluate_known_terms_for_basis(
                mu,
                basis,
                constant=self._last_constant,
                shifts=self._last_shifts,
                residues=self._last_residues,
                derivative=False,
                lu_cache=self._last_lu_cache if self._last_mu == float(mu) else None,
            )
            if lu_cache is not None:
                self._last_lu_cache = dict(lu_cache)
            return block
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
