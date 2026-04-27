from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.core.matrix import as_sparse, is_sparse_like, sparse_linalg_module
from meanfi.integrate.occupations import fermi_dirac

from .base import ChebyshevFOE, RationalFOE
from .common import gershgorin_bounds, matrix_action


_DN_DMU_ABS_FLOOR = 1e-6


def _fermi_derivative(energies: np.ndarray, *, kT: float, mu: float) -> np.ndarray:
    occupation = fermi_dirac(energies, kT, mu)
    return occupation * (1.0 - occupation) / kT


def _scalar_derivative_converged(full: float, half: float, *, rtol: float) -> bool:
    scale = max(abs(full), abs(half), _DN_DMU_ABS_FLOOR)
    if abs(full) <= _DN_DMU_ABS_FLOOR and abs(half) <= _DN_DMU_ABS_FLOOR:
        return True
    return abs(full - half) <= float(rtol) * scale


def _chebyshev_coefficients(
    order: int,
    *,
    center: float,
    scale: float,
    kT: float,
    mu: float,
    oversampling: int,
    derivative: bool,
) -> np.ndarray:
    n_nodes = max(64, int(oversampling) * (order + 1))
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    energies = scale * np.cos(theta) + center
    if derivative:
        values = _fermi_derivative(energies, kT=kT, mu=mu)
    else:
        values = fermi_dirac(energies, kT, mu)

    coeffs = np.empty(order + 1, dtype=float)
    coeffs[0] = np.mean(values)
    for mode in range(1, order + 1):
        coeffs[mode] = 2.0 * np.mean(values * np.cos(mode * theta))
    return coeffs


def _bernstein_positive_poles(max_poles: int) -> np.ndarray:
    rho = 1.0 + 2.0 / max(float(max_poles), 1.0)
    theta = np.pi * (np.arange(max_poles, dtype=float) + 0.5) / max_poles
    return 0.5 * (
        rho * np.exp(1j * theta) + rho ** (-1.0) * np.exp(-1j * theta)
    )


def _fit_rational_coefficients(
    poles: np.ndarray,
    *,
    center: float,
    scale: float,
    kT: float,
    mu: float,
    derivative: bool,
) -> tuple[complex, np.ndarray, np.ndarray]:
    n_columns = 1 + 2 * poles.size
    n_samples = max(96, 8 * n_columns)
    theta = np.pi * (np.arange(n_samples) + 0.5) / n_samples
    nodes = np.cos(theta)
    energies = scale * nodes + center
    if derivative:
        values = _fermi_derivative(energies, kT=kT, mu=mu)
    else:
        values = fermi_dirac(energies, kT, mu)

    design = np.empty((n_samples, n_columns), dtype=complex)
    design[:, 0] = 1.0
    for index, pole in enumerate(poles):
        design[:, 1 + index] = 1.0 / (nodes - pole)
        design[:, 1 + poles.size + index] = 1.0 / (nodes - pole.conjugate())

    coeffs, *_ = np.linalg.lstsq(design, values.astype(complex), rcond=None)
    return coeffs[0], coeffs[1 : 1 + poles.size], coeffs[1 + poles.size :]


class PreparedNormalChebyshevNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        options: ChebyshevFOE,
        tolerance: float,
    ) -> None:
        self.matrix = matrix
        self.kT = float(kT)
        self.options = options
        self.tolerance = float(tolerance)
        self.size = int(getattr(matrix, "shape")[0])

        lower, upper = gershgorin_bounds(matrix)
        width = max(upper - lower, 1e-12)
        self.scale = 0.5 * width * (1.0 + options.spectral_padding)
        self.center = 0.5 * (upper + lower)

        identity = np.eye(self.size, dtype=complex)
        self._basis_terms: list[np.ndarray] = [identity]
        self._trace_moments: list[complex] = [complex(self.size)]
        self._charge_cache: dict[float, tuple[float, float, int, np.ndarray]] = {}
        self._density_cache: dict[float, tuple[np.ndarray, int, np.ndarray]] = {}

    def _apply_rescaled(self, block: np.ndarray) -> np.ndarray:
        return (matrix_action(self.matrix, block) - self.center * block) / self.scale

    def _ensure_order(self, order: int) -> None:
        while len(self._basis_terms) <= order:
            mode = len(self._basis_terms)
            if mode == 1:
                term = self._apply_rescaled(self._basis_terms[0])
            else:
                term = 2.0 * self._apply_rescaled(self._basis_terms[-1]) - self._basis_terms[-2]
            self._basis_terms.append(np.asarray(term, dtype=complex))
            self._trace_moments.append(complex(np.trace(self._basis_terms[-1])))

    def _charge_from_coeffs(self, coeffs: np.ndarray) -> float:
        moments = np.asarray(self._trace_moments[: coeffs.size], dtype=complex)
        return float(np.real(np.dot(coeffs.astype(complex), moments)))

    def _density_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        result = np.zeros_like(self._basis_terms[0], dtype=complex)
        for coeff, term in zip(coeffs, self._basis_terms[: coeffs.size], strict=True):
            result += coeff * term
        return 0.5 * (result + result.conj().T)

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, derivative, _order, _coeffs = self._charge_cache[mu]
            return charge, derivative

        half_order = int(self.options.initial_order)
        self._ensure_order(half_order)
        half_coeffs = _chebyshev_coefficients(
            half_order,
            center=self.center,
            scale=self.scale,
            kT=self.kT,
            mu=mu,
            oversampling=self.options.coefficient_oversampling,
            derivative=False,
        )
        half_derivative_coeffs = _chebyshev_coefficients(
            half_order,
            center=self.center,
            scale=self.scale,
            kT=self.kT,
            mu=mu,
            oversampling=self.options.coefficient_oversampling,
            derivative=True,
        )
        half_charge = self._charge_from_coeffs(half_coeffs)
        half_derivative = self._charge_from_coeffs(half_derivative_coeffs)

        while True:
            order = min(int(self.options.max_order), 2 * half_order)
            self._ensure_order(order)
            full_coeffs = _chebyshev_coefficients(
                order,
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                oversampling=self.options.coefficient_oversampling,
                derivative=False,
            )
            full_derivative_coeffs = _chebyshev_coefficients(
                order,
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                oversampling=self.options.coefficient_oversampling,
                derivative=True,
            )
            full_charge = self._charge_from_coeffs(full_coeffs)
            full_derivative = self._charge_from_coeffs(full_derivative_coeffs)

            if (
                abs(full_charge - half_charge) <= self.tolerance
                and _scalar_derivative_converged(
                    full_derivative,
                    half_derivative,
                    rtol=self.options.dn_dmu_rtol,
                )
            ):
                self._charge_cache[mu] = (
                    full_charge,
                    full_derivative,
                    order,
                    full_coeffs,
                )
                return full_charge, full_derivative

            if order == int(self.options.max_order):
                raise ValueError("Chebyshev FOE did not converge within max_order")

            half_order = order
            half_charge = full_charge
            half_derivative = full_derivative

    def density(self, mu: float) -> np.ndarray:
        if mu in self._density_cache:
            density, _order, _coeffs = self._density_cache[mu]
            return density

        charge_cache = self._charge_cache.get(mu)
        half_order = (
            max(int(self.options.initial_order), charge_cache[2])
            if charge_cache is not None
            else int(self.options.initial_order)
        )
        self._ensure_order(half_order)
        half_coeffs = (
            charge_cache[3]
            if charge_cache is not None and charge_cache[2] == half_order
            else _chebyshev_coefficients(
                half_order,
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                oversampling=self.options.coefficient_oversampling,
                derivative=False,
            )
        )
        half_density = self._density_from_coeffs(half_coeffs)

        while True:
            order = min(int(self.options.max_order), 2 * half_order)
            self._ensure_order(order)
            full_coeffs = _chebyshev_coefficients(
                order,
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                oversampling=self.options.coefficient_oversampling,
                derivative=False,
            )
            full_density = self._density_from_coeffs(full_coeffs)

            if float(np.max(np.abs(full_density - half_density))) <= self.tolerance:
                self._density_cache[mu] = (full_density, order, full_coeffs)
                return full_density

            if order == int(self.options.max_order):
                raise ValueError("Chebyshev FOE did not converge within max_order")

            half_order = order
            half_density = full_density


class PreparedNormalRationalNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        options: RationalFOE,
        tolerance: float,
    ) -> None:
        self.matrix = matrix
        self.kT = float(kT)
        self.options = options
        self.tolerance = float(tolerance)
        self.size = int(getattr(matrix, "shape")[0])
        lower, upper = gershgorin_bounds(matrix)
        width = max(upper - lower, 1e-12)
        self.scale = 0.5 * width
        self.center = 0.5 * (upper + lower)

        self._identity = np.eye(self.size, dtype=complex)
        self._positive_poles = _bernstein_positive_poles(int(options.max_poles))
        self._prepared_poles = 0
        self._resolvents: list[np.ndarray] = []
        self._traces: list[complex] = []
        self._charge_cache: dict[float, tuple[float, float, int, tuple[complex, np.ndarray, np.ndarray]]] = {}
        self._density_cache: dict[float, tuple[np.ndarray, int, tuple[complex, np.ndarray, np.ndarray]]] = {}

    def _inverse_resolvent(self, pole: complex) -> np.ndarray:
        shift = self.center + self.scale * pole
        rhs = self.scale * self._identity
        if is_sparse_like(self.matrix):
            shifted = as_sparse(self.matrix).tocsc().astype(complex)
            shifted = shifted.copy()
            diagonal = np.asarray(shifted.diagonal(), dtype=complex) - shift
            shifted.setdiag(diagonal)
            lu = sparse_linalg_module().splu(shifted)
            return np.asarray(lu.solve(rhs), dtype=complex)

        dense = np.asarray(self.matrix, dtype=complex)
        shifted = dense - shift * np.eye(self.size, dtype=complex)
        return np.asarray(np.linalg.solve(shifted, rhs), dtype=complex)

    def _ensure_poles(self, pole_count: int) -> None:
        while self._prepared_poles < pole_count:
            pole = self._positive_poles[self._prepared_poles]
            resolvent = self._inverse_resolvent(pole)
            self._resolvents.append(resolvent)
            self._traces.append(complex(np.trace(resolvent)))
            self._prepared_poles += 1

    def _charge_from_coeffs(
        self,
        coeffs: tuple[complex, np.ndarray, np.ndarray],
    ) -> float:
        constant, positive_coeffs, negative_coeffs = coeffs
        traces = np.asarray(self._traces[: positive_coeffs.size], dtype=complex)
        total = constant * self.size
        if traces.size:
            total = total + np.dot(positive_coeffs, traces)
            total = total + np.dot(negative_coeffs, np.conjugate(traces))
        return float(np.real(total))

    def _density_from_coeffs(
        self,
        coeffs: tuple[complex, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        constant, positive_coeffs, negative_coeffs = coeffs
        result = constant * self._identity.astype(complex, copy=True)
        for coeff, resolvent in zip(
            positive_coeffs,
            self._resolvents[: positive_coeffs.size],
            strict=True,
        ):
            result += coeff * resolvent
        for coeff, resolvent in zip(
            negative_coeffs,
            self._resolvents[: negative_coeffs.size],
            strict=True,
        ):
            result += coeff * resolvent.conj().T
        return 0.5 * (result + result.conj().T)

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, derivative, _order, _coeffs = self._charge_cache[mu]
            return charge, derivative

        half_poles = int(self.options.initial_poles)
        self._ensure_poles(half_poles)
        half_coeffs = _fit_rational_coefficients(
            self._positive_poles[:half_poles],
            center=self.center,
            scale=self.scale,
            kT=self.kT,
            mu=mu,
            derivative=False,
        )
        half_derivative_coeffs = _fit_rational_coefficients(
            self._positive_poles[:half_poles],
            center=self.center,
            scale=self.scale,
            kT=self.kT,
            mu=mu,
            derivative=True,
        )
        half_charge = self._charge_from_coeffs(half_coeffs)
        half_derivative = self._charge_from_coeffs(half_derivative_coeffs)

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            self._ensure_poles(pole_count)
            full_coeffs = _fit_rational_coefficients(
                self._positive_poles[:pole_count],
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                derivative=False,
            )
            full_derivative_coeffs = _fit_rational_coefficients(
                self._positive_poles[:pole_count],
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                derivative=True,
            )
            full_charge = self._charge_from_coeffs(full_coeffs)
            full_derivative = self._charge_from_coeffs(full_derivative_coeffs)

            if (
                abs(full_charge - half_charge) <= self.tolerance
                and _scalar_derivative_converged(
                    full_derivative,
                    half_derivative,
                    rtol=self.options.dn_dmu_rtol,
                )
            ):
                self._charge_cache[mu] = (
                    full_charge,
                    full_derivative,
                    pole_count,
                    full_coeffs,
                )
                return full_charge, full_derivative

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_charge = full_charge
            half_derivative = full_derivative

    def density(self, mu: float) -> np.ndarray:
        if mu in self._density_cache:
            density, _order, _coeffs = self._density_cache[mu]
            return density

        charge_cache = self._charge_cache.get(mu)
        half_poles = (
            max(int(self.options.initial_poles), charge_cache[2])
            if charge_cache is not None
            else int(self.options.initial_poles)
        )
        self._ensure_poles(half_poles)
        half_coeffs = (
            charge_cache[3]
            if charge_cache is not None and charge_cache[2] == half_poles
            else _fit_rational_coefficients(
                self._positive_poles[:half_poles],
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                derivative=False,
            )
        )
        half_density = self._density_from_coeffs(half_coeffs)

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            self._ensure_poles(pole_count)
            full_coeffs = _fit_rational_coefficients(
                self._positive_poles[:pole_count],
                center=self.center,
                scale=self.scale,
                kT=self.kT,
                mu=mu,
                derivative=False,
            )
            full_density = self._density_from_coeffs(full_coeffs)

            if float(np.max(np.abs(full_density - half_density))) <= self.tolerance:
                self._density_cache[mu] = (full_density, pole_count, full_coeffs)
                return full_density

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_density = full_density
