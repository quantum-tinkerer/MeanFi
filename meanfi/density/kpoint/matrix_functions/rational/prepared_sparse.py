from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.errors import ConvergenceError

from meanfi.tb.ops import as_sparse, is_sparse_like

from ..base import RationalFOE
from meanfi.density.kpoint.occupations import fermi_dirac
from ..common import spectral_interval, shift_by_mu
from ..mumps_backend import SelectedInverseFactorization, build_selected_inverse_pattern
from .common import (
    SparseRationalLayout,
    SparseRationalTerms,
    _sparse_shifted_matrix,
)
from .scheme import (
    _AAAIntervalCacheEntry,
    _aaa_sample_grid,
    _aaa_terms_for_interval,
    thermal_errors,
    fit_entropy,
)


class PreparedMumpsRationalNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        q_diag: np.ndarray,
        options: RationalFOE,
        layout: SparseRationalLayout,
        matrix_function_tol: float,
        compute_entropy: bool = False,
        workspace_dtype: np.dtype = np.dtype(complex),
        shared_aaa_interval_cache: list[_AAAIntervalCacheEntry] | None = None,
    ) -> None:
        if not is_sparse_like(matrix):
            raise ValueError("Sparse MUMPS-backed RationalFOE requires sparse matrices")

        self.workspace_dtype = np.dtype(workspace_dtype)
        self.matrix = as_sparse(matrix).astype(self.workspace_dtype).tocsr()
        self.kT = float(kT)
        self.q_diag = np.asarray(q_diag, dtype=float)
        self.options = options
        self.matrix_function_tol = float(matrix_function_tol)
        self.compute_entropy = compute_entropy
        self.layout = layout
        self.size = int(getattr(matrix, "shape")[0])
        if layout.charge.size != self.size:
            raise ValueError("Sparse layout must match the Hamiltonian size")
        self._aaa_interval_cache: list[_AAAIntervalCacheEntry] = (
            shared_aaa_interval_cache if shared_aaa_interval_cache is not None else []
        )
        self.matrix_function_error: float | None = None
        self._last_mu: float | None = None
        self._last_terms: SparseRationalTerms | None = None
        self._last_factorizations: dict[complex, SelectedInverseFactorization] = {}
        self._last_inverse_entries: dict[complex, np.ndarray] = {}
        self._entry_positions: dict[int, int] = {}

    def _sparse_terms(self, mu: float) -> SparseRationalTerms:
        pole_count = self.options.max_poles
        shifted = shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype)
        lower, upper = spectral_interval(shifted)
        padding = 1e-12 * max(1.0, float(upper - lower))
        lower, upper = lower - padding, upper + padding
        tolerance = self.matrix_function_tol
        margin = 0.0
        for entry in self._aaa_interval_cache:
            contained = entry.lower <= lower and upper <= entry.upper
            if (
                entry.kT == self.kT
                and not contained
                and max(lower, entry.lower) < min(upper, entry.upper)
            ):
                # A nearby miss gets room for subsequent k-point and mu shifts.
                # Keep the first fit narrow so one-off evaluations pay no premium.
                margin = 0.2 * (upper - lower)
            if (
                entry.kT != self.kT
                or not contained
                or entry.terms.pole_count > pole_count
            ):
                continue
            grid = _aaa_sample_grid(
                lower, upper, kT=self.kT, count=max(2048, 32 * pole_count)
            )
            targets = fermi_dirac(grid, self.kT, 0.0)[:, None]
            error = float(thermal_errors(entry.terms, grid, targets)[0])
            if error <= tolerance:
                self.matrix_function_error = error
                terms = self._with_entropy(entry.terms, entry.lower, entry.upper)
                self._aaa_interval_cache[:] = [
                    _AAAIntervalCacheEntry(entry.lower, entry.upper, self.kT, terms)
                ]
                return terms
        # Widening is optional: a restricted pole budget may only fit the actual
        # spectrum. In that case retry its original interval before failing.
        for extra in (margin, 0.0) if margin else (0.0,):
            fit_lower, fit_upper = lower - extra, upper + extra
            try:
                terms = _aaa_terms_for_interval(
                    pole_cap=pole_count,
                    lower=fit_lower,
                    upper=fit_upper,
                    kT=self.kT,
                    initial_poles=self.options.initial_poles,
                    scalar_tolerance=tolerance,
                )
                break
            except ValueError as exc:
                if extra == 0.0:
                    raise ConvergenceError(str(exc)) from exc
        self.matrix_function_error = terms.error
        terms = self._with_entropy(terms, fit_lower, fit_upper)
        # Keep one scalar fit shared across k-points, never their factorizations.
        self._aaa_interval_cache[:] = [
            _AAAIntervalCacheEntry(fit_lower, fit_upper, self.kT, terms)
        ]
        return terms

    def _with_entropy(self, terms, lower, upper):
        if self.compute_entropy and terms.entropy_residues is None:
            return fit_entropy(terms, lower=lower, upper=upper, kT=self.kT)
        return terms

    def _prepare(self, mu: float) -> None:
        if self._last_mu == float(mu):
            return
        terms = self._sparse_terms(mu)
        shifted = shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype)
        self._last_factorizations = {}
        self._last_inverse_entries = {}
        self._entry_positions.clear()
        for shift in terms.shifts:
            key = complex(shift)
            factorization = SelectedInverseFactorization()
            factorization.factor(_sparse_shifted_matrix(shifted, key))
            self._last_factorizations[key] = factorization
            self._last_inverse_entries[key] = np.empty(0, dtype=complex)
        self._last_terms = terms
        self._last_mu = float(mu)

    def _select(self, mu, pattern):
        """Evaluate only missing inverse entries; store only evaluated entries."""
        self._prepare(mu)
        keys = self.size * pattern.cols + pattern.rows
        positions = np.array([self._entry_positions.get(int(key), -1) for key in keys])
        missing = positions < 0
        if np.any(missing):
            if not np.all(missing):
                pattern = build_selected_inverse_pattern(
                    size=self.size,
                    rows=pattern.rows[missing],
                    cols=pattern.cols[missing],
                )
            start = len(self._entry_positions)
            positions[missing] = np.arange(start, start + np.count_nonzero(missing))
            self._entry_positions.update(
                zip(keys[missing], positions[missing], strict=True)
            )
            for shift, factorization in self._last_factorizations.items():
                self._last_inverse_entries[shift] = np.r_[
                    self._last_inverse_entries[shift],
                    factorization.selected_inverse(pattern),
                ]
        return {
            shift: values[positions]
            for shift, values in self._last_inverse_entries.items()
        }

    def charge(self, mu: float) -> float:
        """Evaluate the physical charge diagonal without energy-only entries."""
        entries = self._select(mu, self.layout.charge)
        terms = self._last_terms
        return self.layout.charge_from_inverse_entries(
            entries,
            constant=terms.constant,
            shifts=terms.shifts,
            residues=terms.residues,
        )

    def density_values(self, mu: float) -> np.ndarray:
        """Evaluate requested density entries, without requiring a charge probe."""
        if self.layout.value_positions.size == 0:
            return np.empty(0, dtype=complex)
        entries = self._select(mu, self.layout.density)
        terms = self._last_terms
        return self.layout.density_values_from_inverse_entries(
            entries,
            constant=terms.constant,
            shifts=terms.shifts,
            residues=terms.residues,
        )

    def thermodynamics(self, mu: float) -> tuple[float, float | None]:
        """Return Tr[H f(H-mu Q)] and entropy, obtaining missing diagonals if needed.

        BdG particle/hole normalization belongs to the caller. The identity
        H(A-zI)^-1 = I + (zI + mu Q)(A-zI)^-1 supplies the energy without
        additional factorizations; entropy uses the same poles with fitted residues.
        """
        entries = self._select(mu, self.layout.diagonal)
        terms = self._last_terms
        energy = float(np.real(terms.constant * np.sum(self.matrix.diagonal())))
        entropy = (
            float(np.real(terms.entropy_constant * self.size))
            if self.compute_entropy
            else None
        )
        for index, (shift, residue) in enumerate(
            zip(terms.shifts, terms.residues, strict=True)
        ):
            diagonal = entries[complex(shift)]
            trace_inverse = np.sum(diagonal)
            energy += float(
                2.0
                * np.real(
                    residue
                    * (
                        self.size
                        + shift * trace_inverse
                        + float(mu) * (self.q_diag @ diagonal)
                    )
                )
            )
            if self.compute_entropy:
                entropy += float(
                    2.0 * np.real(terms.entropy_residues[index] * trace_inverse)
                )
        return energy, entropy
