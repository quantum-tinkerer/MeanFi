from __future__ import annotations

from typing import Any

import numpy as np

from .base import ChebyshevFOE
from .common import (
    chebyshev_foe_coefficients,
    matrix_action,
    scalar_derivative_converged,
    spectral_interval,
    trace_probe_block,
    weighted_trace_from_basis_result,
    workspace_matrix,
)
from .chebyshev import _chebyshev_density_block_at_order
from .common import shift_by_mu


class PreparedNormalChebyshevNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        options: ChebyshevFOE,
        charge_tolerance: float | None,
        workspace_dtype: np.dtype = np.dtype(complex),
        trace_weights_diag: np.ndarray | None = None,
    ) -> None:
        self.workspace_dtype = np.dtype(workspace_dtype)
        self.matrix = workspace_matrix(matrix, self.workspace_dtype)
        self.kT = float(kT)
        self.options = options
        self.charge_tolerance = (
            None if charge_tolerance is None else float(charge_tolerance)
        )
        self.size = int(getattr(matrix, "shape")[0])

        lower, upper = spectral_interval(
            matrix,
            spectral_padding=options.spectral_padding,
        )
        width = max(upper - lower, 1e-12)
        self.scale = 0.5 * width
        self.center = 0.5 * (upper + lower)

        self._identity = np.eye(self.size, dtype=self.workspace_dtype)
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

        self._trace_moments: list[complex] = [
            complex(
                weighted_trace_from_basis_result(
                    self._trace_basis,
                    self._trace_basis,
                    weights_diag=self._trace_weights,
                    estimator=self._trace_estimator,
                    trace_columns=self._trace_columns,
                )
            )
        ]
        self._prepared_order = 0
        self._tail_prev = np.array(self._trace_basis, copy=True)
        self._tail_curr: np.ndarray | None = None
        self._charge_cache: dict[float, tuple[float, float, int]] = {}
        self._coefficient_cache: dict[tuple[float, int, bool], np.ndarray] = {}

    def _apply_rescaled(self, block: np.ndarray) -> np.ndarray:
        return (matrix_action(self.matrix, block) - self.center * block) / self.scale

    def ensure_trace_order(self, order: int) -> None:
        while self._prepared_order < order:
            if self._prepared_order == 0:
                next_term = self._apply_rescaled(self._tail_prev)
            else:
                assert self._tail_curr is not None
                next_term = 2.0 * self._apply_rescaled(self._tail_curr) - self._tail_prev
                self._tail_prev = self._tail_curr

            self._tail_curr = np.asarray(next_term, dtype=self.workspace_dtype)
            moment = weighted_trace_from_basis_result(
                self._tail_curr,
                self._trace_basis,
                weights_diag=self._trace_weights,
                estimator=self._trace_estimator,
                trace_columns=self._trace_columns,
            )
            self._trace_moments.append(complex(moment))
            self._prepared_order += 1

    def _charge_from_coeffs(self, coeffs: np.ndarray) -> float:
        moments = np.asarray(self._trace_moments[: coeffs.size], dtype=complex)
        return float(np.real(np.dot(coeffs.astype(complex), moments)))

    def _coefficients(self, mu: float, order: int, *, derivative: bool) -> np.ndarray:
        cache_key = (float(mu), int(order), bool(derivative))
        cached = self._coefficient_cache.get(cache_key)
        if cached is not None:
            return cached

        coeffs = chebyshev_foe_coefficients(
            order,
            center=self.center,
            scale=self.scale,
            kT=self.kT,
            mu=mu,
            oversampling=self.options.coefficient_oversampling,
            derivative=derivative,
        )
        self._coefficient_cache[cache_key] = coeffs
        return coeffs

    def _density_columns_from_coeffs(
        self,
        coeffs: np.ndarray,
        basis: np.ndarray,
    ) -> np.ndarray:
        result = coeffs[0] * np.array(basis, copy=True)
        if coeffs.size == 1:
            return result

        previous = np.asarray(basis, dtype=self.workspace_dtype)
        current = np.asarray(self._apply_rescaled(previous), dtype=self.workspace_dtype)
        result += coeffs[1] * current

        for mode in range(2, coeffs.size):
            next_term = 2.0 * self._apply_rescaled(current) - previous
            next_term = np.asarray(next_term, dtype=self.workspace_dtype)
            result += coeffs[mode] * next_term
            previous, current = current, next_term

        return result

    def _density_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        columns = self._density_columns_from_coeffs(coeffs, self._identity)
        return 0.5 * (columns + columns.conj().T)

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, derivative, _order = self._charge_cache[mu]
            return charge, derivative
        if self.charge_tolerance is None:
            raise RuntimeError("Charge tolerance is not configured for this Chebyshev node")

        half_order = int(self.options.initial_order)
        self.ensure_trace_order(half_order)
        half_coeffs = self._coefficients(mu, half_order, derivative=False)
        half_derivative_coeffs = self._coefficients(mu, half_order, derivative=True)
        half_charge = self._charge_from_coeffs(half_coeffs)
        half_derivative = self._charge_from_coeffs(half_derivative_coeffs)

        while True:
            order = min(int(self.options.max_order), 2 * half_order)
            self.ensure_trace_order(order)
            full_coeffs = self._coefficients(mu, order, derivative=False)
            full_derivative_coeffs = self._coefficients(mu, order, derivative=True)
            full_charge = self._charge_from_coeffs(full_coeffs)
            full_derivative = self._charge_from_coeffs(full_derivative_coeffs)

            if (
                abs(full_charge - half_charge) <= self.charge_tolerance
                and scalar_derivative_converged(
                    full_derivative,
                    half_derivative,
                    rtol=self.options.dn_dmu_rtol,
                )
            ):
                self._charge_cache[mu] = (full_charge, full_derivative, order)
                return full_charge, full_derivative

            if order == int(self.options.max_order):
                raise ValueError("Chebyshev FOE did not converge within max_order")

            half_order = order
            half_charge = full_charge
            half_derivative = full_derivative

    def density_from_charge_order(self, mu: float) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError("Charge order is unavailable for the requested chemical potential")

        coeffs = self._coefficients(mu, charge_cache[2], derivative=False)
        return self._density_from_coeffs(coeffs)

    def density_columns_from_charge_order(self, mu: float, basis: np.ndarray) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError("Charge order is unavailable for the requested chemical potential")
        coeffs = self._coefficients(mu, charge_cache[2], derivative=False)
        return self._density_columns_from_coeffs(
            coeffs,
            np.asarray(basis, dtype=self.workspace_dtype),
        )

    def density(self, mu: float, *, tolerance: float) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        half_order = (
            max(int(self.options.initial_order), charge_cache[2])
            if charge_cache is not None
            else int(self.options.initial_order)
        )
        half_coeffs = self._coefficients(mu, half_order, derivative=False)
        half_density = self._density_from_coeffs(half_coeffs)

        while True:
            order = min(int(self.options.max_order), 2 * half_order)
            full_coeffs = self._coefficients(mu, order, derivative=False)
            full_density = self._density_from_coeffs(full_coeffs)

            if float(np.max(np.abs(full_density - half_density))) <= float(tolerance):
                return full_density

            if order == int(self.options.max_order):
                raise ValueError("Chebyshev FOE did not converge within max_order")

            half_order = order
            half_density = full_density


class PreparedShiftedChebyshevNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        q_diag: np.ndarray,
        options: ChebyshevFOE,
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

    def _trace_scalar(self, block: np.ndarray) -> float:
        return weighted_trace_from_basis_result(
            block,
            self._trace_basis,
            weights_diag=self._trace_weights,
            estimator=self._trace_estimator,
            trace_columns=self._trace_columns,
        )

    def _evaluate_at_order(
        self,
        mu: float,
        basis: np.ndarray,
        *,
        order: int,
        derivative: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        return _chebyshev_density_block_at_order(
            shifted,
            np.asarray(basis, dtype=self.workspace_dtype),
            kT=self.kT,
            q_diag=self.q_diag,
            order=order,
            options=self.options,
            derivative=derivative,
            workspace_dtype=self.workspace_dtype,
        )

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, derivative, _order = self._charge_cache[mu]
            return charge, derivative

        half_order = int(self.options.initial_order)
        half_block, half_derivative_block = self._evaluate_at_order(
            mu,
            self._trace_basis,
            order=half_order,
            derivative=True,
        )
        half_charge = self._trace_scalar(half_block)
        half_derivative = (
            0.0 if half_derivative_block is None else self._trace_scalar(half_derivative_block)
        )

        while True:
            order = min(int(self.options.max_order), 2 * half_order)
            full_block, full_derivative_block = self._evaluate_at_order(
                mu,
                self._trace_basis,
                order=order,
                derivative=True,
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
                self._charge_cache[mu] = (full_charge, full_derivative, order)
                return full_charge, full_derivative

            if order == int(self.options.max_order):
                raise ValueError("Chebyshev FOE did not converge within max_order")

            half_order = order
            half_charge = full_charge
            half_derivative = full_derivative

    def density_columns_from_charge_order(self, mu: float, basis: np.ndarray) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError("Charge order is unavailable for the requested chemical potential")
        block, _ = self._evaluate_at_order(
            mu,
            basis,
            order=charge_cache[2],
            derivative=False,
        )
        return block

    def density_from_charge_order(self, mu: float) -> np.ndarray:
        block = self.density_columns_from_charge_order(
            mu,
            np.eye(self.size, dtype=self.workspace_dtype),
        )
        return 0.5 * (block + block.conj().T)
