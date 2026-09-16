from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.errors import ConvergenceError

from meanfi.tb.ops import as_sparse, is_sparse_like

from ..base import RationalFOE
from meanfi.density.kpoint.occupations import fermi_dirac
from ..common import spectral_interval, shift_by_mu
from ..mumps_backend import SelectedInverseFactorization
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
        charge_tolerance: float | None,
        layout: SparseRationalLayout,
        matrix_function_tol: float,
        compute_thermodynamics: bool = False,
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
        self.charge_tolerance = (
            None if charge_tolerance is None else float(charge_tolerance)
        )
        self.matrix_function_tol = float(matrix_function_tol)
        self.compute_thermodynamics = compute_thermodynamics
        self.layout = layout
        self.size = int(getattr(matrix, "shape")[0])
        if layout.charge.size != self.size:
            raise ValueError("Sparse layout must match the Hamiltonian size")
        if compute_thermodynamics and layout.charge.nnz != self.size:
            raise ValueError("Thermodynamics requires all inverse diagonal entries")
        self._aaa_interval_cache: list[_AAAIntervalCacheEntry] = (
            shared_aaa_interval_cache if shared_aaa_interval_cache is not None else []
        )
        self.matrix_function_error: float | None = None
        self._last_mu: float | None = None
        self._last_charge: float | None = None
        self._last_terms: SparseRationalTerms | None = None
        self._last_factorizations: dict[complex, SelectedInverseFactorization] = {}
        self._last_charge_entries: dict[complex, np.ndarray] = {}

    def _charge_scalar_tolerance(self) -> float:
        weight_sum = float(np.sum(np.abs(self.layout.charge_weights)))
        if self.charge_tolerance is None or weight_sum <= 0.0:
            return self.matrix_function_tol
        # Charge is a weighted trace, so the scalar Fermi-operator error must
        # shrink with the total trace weight to keep the filling solve stable.
        return min(self.matrix_function_tol, self.charge_tolerance / weight_sum)

    def _sparse_terms(self, mu: float) -> SparseRationalTerms:
        pole_count = self.options.max_poles
        shifted = shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype)
        lower, upper = spectral_interval(shifted)
        padding = 1e-12 * max(1.0, float(upper - lower))
        lower, upper = lower - padding, upper + padding
        tolerance = self._charge_scalar_tolerance()
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
        if self.compute_thermodynamics and terms.entropy_residues is None:
            return fit_entropy(terms, lower=lower, upper=upper, kT=self.kT)
        return terms

    def charge(self, mu: float) -> float:
        """Evaluate charge, retaining this point's factors for density and energy."""
        if self._last_mu == float(mu):
            return self._last_charge
        terms = self._sparse_terms(mu)
        shifted = shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype)
        factorizations = {}
        entries = {}
        for shift in terms.shifts:
            key = complex(shift)
            factorization = SelectedInverseFactorization()
            factorization.factor(_sparse_shifted_matrix(shifted, key))
            factorizations[key] = factorization
            entries[key] = factorization.selected_inverse(self.layout.charge)
        charge = self.layout.charge_from_inverse_entries(
            entries,
            constant=terms.constant,
            shifts=terms.shifts,
            residues=terms.residues,
        )
        self._last_charge = charge
        self._last_terms = terms
        self._last_factorizations = factorizations
        self._last_charge_entries = entries
        self._last_mu = float(mu)
        return self._last_charge

    def density_values_from_charge_order(self, mu: float) -> np.ndarray:
        """Use only the factors from charge evaluation at this same mu."""
        if self._last_mu != float(mu) or self._last_terms is None:
            raise ValueError("Evaluate charge at the requested mu before density")
        terms = self._last_terms
        layout = self.layout
        entries = {}
        for shift in terms.shifts:
            key = complex(shift)
            values = np.empty(layout.density.nnz, dtype=np.complex128)
            values[layout.density_charge_positions] = self._last_charge_entries[key][
                layout.charge_positions
            ]
            if layout.extra.nnz:
                values[layout.density_extra_positions] = self._last_factorizations[
                    key
                ].selected_inverse(layout.extra)
            entries[key] = values
        return layout.density_values_from_inverse_entries(
            entries,
            constant=terms.constant,
            shifts=terms.shifts,
            residues=terms.residues,
        )

    def thermodynamics(self, mu: float) -> tuple[float, float]:
        """Return Tr[H f(H-mu Q)] and entropy from the retained inverse diagonals.

        BdG particle/hole normalization belongs to the caller. The identity
        H(A-zI)^-1 = I + (zI + mu Q)(A-zI)^-1 supplies the energy without
        additional factorizations; entropy uses the same poles with fitted residues.
        """
        if self._last_mu != float(mu) or self._last_terms is None:
            raise ValueError(
                "Evaluate charge at the requested mu before thermodynamics"
            )
        terms = self._last_terms
        if not self.compute_thermodynamics:
            raise ValueError("Prepare the node with compute_thermodynamics=True first")
        energy = float(np.real(terms.constant * np.sum(self.matrix.diagonal())))
        entropy = float(np.real(terms.entropy_constant * self.size))
        for shift, residue, entropy_residue in zip(
            terms.shifts, terms.residues, terms.entropy_residues, strict=True
        ):
            diagonal = self._last_charge_entries[complex(shift)]
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
            entropy += float(2.0 * np.real(entropy_residue * trace_inverse))
        return energy, entropy
