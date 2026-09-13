from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import as_sparse, is_sparse_like

from ..base import RationalFOE
from ..common import spectral_interval, shift_by_mu
from ..mumps_backend import (
    SelectedInverseFactorization,
    build_selected_inverse_pattern,
)
from ...occupations import fermi_dirac
from .common import (
    SparseRationalTerms,
    _pattern_subset_mappings,
    _sparse_shifted_matrix,
    build_sparse_charge_pattern,
    build_sparse_density_pattern,
)
from .scheme import (
    _AAAIntervalCacheEntry,
    _aaa_sample_grid,
    _aaa_terms_for_interval,
    _barycentric_evaluate,
    _evaluate_canonical_rational,
    _ozaki_terms,
)


class PreparedMumpsRationalNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        q_diag: np.ndarray,
        options: RationalFOE,
        charge_tolerance: float,
        density_coordinates: DensityCoordinates,
        density_tolerance: float,
        workspace_dtype: np.dtype = np.dtype(complex),
        trace_weights_diag: np.ndarray | None = None,
        shared_aaa_interval_cache: list[_AAAIntervalCacheEntry] | None = None,
    ) -> None:
        if options.rational_scheme not in {"ozaki", "aaa"}:
            raise ValueError(
                "Sparse RationalFOE currently requires rational_scheme='ozaki' or 'aaa'"
            )
        if not is_sparse_like(matrix):
            raise ValueError("Sparse MUMPS-backed RationalFOE requires sparse matrices")

        self.workspace_dtype = np.dtype(workspace_dtype)
        self.matrix = as_sparse(matrix).astype(self.workspace_dtype).tocsr()
        self.kT = float(kT)
        self.q_diag = np.asarray(q_diag, dtype=float)
        self.options = options
        self.charge_tolerance = float(charge_tolerance)
        self.density_tolerance = float(density_tolerance)
        self.size = int(getattr(matrix, "shape")[0])
        self._trace_weights = (
            np.ones(self.size, dtype=float)
            if trace_weights_diag is None
            else np.asarray(trace_weights_diag, dtype=float)
        )
        self._charge_pattern = build_sparse_charge_pattern(self._trace_weights)
        self._density_pattern = build_sparse_density_pattern(
            size=self.size,
            density_coordinates=density_coordinates,
        )
        density_extra_positions = np.asarray(
            [
                position
                for position in range(self._density_pattern.pattern.nnz)
                if (
                    int(self._density_pattern.pattern.rows[position]),
                    int(self._density_pattern.pattern.cols[position]),
                )
                not in self._charge_pattern.pattern.lookup
            ],
            dtype=int,
        )
        self._density_extra_pattern = build_selected_inverse_pattern(
            size=self.size,
            rows=self._density_pattern.pattern.rows[density_extra_positions],
            cols=self._density_pattern.pattern.cols[density_extra_positions],
        )
        (
            self._charge_to_density_source_positions,
            self._charge_to_density_target_positions,
        ) = _pattern_subset_mappings(
            self._charge_pattern.pattern, self._density_pattern.pattern
        )
        (
            self._extra_to_density_source_positions,
            self._extra_to_density_target_positions,
        ) = _pattern_subset_mappings(
            self._density_extra_pattern, self._density_pattern.pattern
        )
        self._aaa_interval_cache: list[_AAAIntervalCacheEntry] = (
            shared_aaa_interval_cache if shared_aaa_interval_cache is not None else []
        )
        self._last_mu: float | None = None
        self._last_charge: float | None = None
        self._last_terms: SparseRationalTerms | None = None
        self._last_factorizations: dict[complex, SelectedInverseFactorization] = {}
        self._last_charge_entries: dict[complex, np.ndarray] = {}

    def _charge_scalar_tolerance(self) -> float:
        weight_sum = float(np.sum(np.abs(self._charge_pattern.charge_weights)))
        if weight_sum <= 0.0:
            return self.density_tolerance
        # Charge is a weighted trace, so the scalar Fermi-operator error must
        # shrink with the total trace weight to keep the filling solve stable.
        return max(
            np.finfo(float).eps,
            min(
                self.density_tolerance,
                float(self.charge_tolerance) / weight_sum,
            ),
        )

    def _aaa_cached_terms_for_interval(
        self,
        *,
        lower: float,
        upper: float,
        pole_cap: int,
        scalar_tolerance: float,
    ) -> SparseRationalTerms | None:
        for entry in self._aaa_interval_cache:
            if not np.isclose(entry.kT, self.kT):
                continue
            if lower < entry.lower or upper > entry.upper:
                continue
            if (
                entry.terms.support_count is not None
                and entry.terms.support_count > pole_cap
            ):
                continue
            certification_grid = _aaa_sample_grid(
                lower,
                upper,
                count=max(1024, 64 * int(pole_cap)),
                kT=self.kT,
            )
            exact = np.asarray(
                fermi_dirac(certification_grid, self.kT, 0.0), dtype=complex
            )
            barycentric = (
                _barycentric_evaluate(
                    certification_grid, entry.support_x, entry.support_y, entry.weights
                )
                if entry.support_x.size
                else np.full(certification_grid.shape, entry.terms.constant)
            )
            pole_values = _evaluate_canonical_rational(
                certification_grid,
                constant=entry.terms.constant,
                shifts=entry.terms.shifts,
                residues=entry.terms.residues,
                tail_lower_bound=entry.terms.tail_lower_bound,
                tail_upper_bound=entry.terms.tail_upper_bound,
            )
            scalar_error = float(np.max(np.abs(exact - pole_values), initial=0.0))
            barycentric_gap = float(
                np.max(np.abs(pole_values - barycentric), initial=0.0)
            )
            if (
                scalar_error <= scalar_tolerance
                and barycentric_gap <= 0.1 * scalar_tolerance
            ):
                return entry.terms
        return None

    def _sparse_terms(
        self,
        mu: float,
        *,
        pole_count: int,
        scalar_tolerance: float | None = None,
    ) -> SparseRationalTerms:
        if self.options.rational_scheme == "ozaki":
            return _ozaki_terms(pole_count, self.kT)
        if scalar_tolerance is None:
            scalar_tolerance = self.density_tolerance
        shifted = shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype)
        lower, upper = spectral_interval(shifted)
        padding = 1e-12 * max(1.0, float(upper - lower))
        lower, upper = lower - padding, upper + padding
        cached = self._aaa_cached_terms_for_interval(
            lower=lower,
            upper=upper,
            pole_cap=pole_count,
            scalar_tolerance=scalar_tolerance,
        )
        if cached is not None:
            return cached
        try:
            terms, builder = _aaa_terms_for_interval(
                pole_cap=pole_count,
                lower=lower,
                upper=upper,
                kT=self.kT,
                initial_poles=self.options.initial_poles,
                scalar_tolerance=scalar_tolerance,
            )
        except ValueError as exc:
            if "AAA scalar certification failed" not in str(exc):
                raise
            raise ValueError("Rational FOE did not converge within max_poles") from exc
        # One reusable scalar fit; no retained matrices or factorizations.
        self._aaa_interval_cache[:] = [
            _AAAIntervalCacheEntry(
                lower=lower,
                upper=upper,
                kT=self.kT,
                support_x=np.asarray(builder.support_x, dtype=float),
                support_y=np.asarray(builder.support_y, dtype=complex),
                weights=np.asarray(builder.weights, dtype=complex),
                terms=terms,
            )
        ]
        return terms

    def _evaluate_charge_for_pole_count(self, mu: float, pole_count: int):
        terms = self._sparse_terms(
            mu,
            pole_count=pole_count,
            scalar_tolerance=self._charge_scalar_tolerance(),
        )
        shifted = shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype)
        factorizations = {}
        entries = {}
        for shift in terms.shifts:
            key = complex(shift)
            factorization = SelectedInverseFactorization()
            factorization.factor(_sparse_shifted_matrix(shifted, key))
            factorizations[key] = factorization
            entries[key] = factorization.selected_inverse(self._charge_pattern.pattern)
        charge = self._charge_pattern.charge_from_inverse_entries(
            entries,
            constant=terms.constant,
            shifts=terms.shifts,
            residues=terms.residues,
        )
        return charge, terms, factorizations, entries

    def charge(self, mu: float) -> float:
        """Evaluate charge, retaining this point's factors for the density pass."""
        if self._last_mu == float(mu):
            return self._last_charge
        if self.options.rational_scheme == "aaa":
            result = self._evaluate_charge_for_pole_count(mu, self.options.max_poles)
        else:
            poles = self.options.initial_poles
            previous = self._evaluate_charge_for_pole_count(mu, poles)[0]
            while True:
                poles = min(self.options.max_poles, 2 * poles)
                result = self._evaluate_charge_for_pole_count(mu, poles)
                if abs(result[0] - previous) <= self.charge_tolerance:
                    break
                if poles == self.options.max_poles:
                    raise ValueError("Rational FOE did not converge within max_poles")
                previous = result[0]
        (
            self._last_charge,
            self._last_terms,
            self._last_factorizations,
            self._last_charge_entries,
        ) = result
        self._last_mu = float(mu)
        return self._last_charge

    def _request_extra_density_entries_from_cached_factorizations(
        self,
        *,
        shifts: np.ndarray,
    ) -> dict[complex, np.ndarray]:
        if self._density_extra_pattern.nnz == 0:
            return {
                complex(shift): np.empty(0, dtype=np.complex128) for shift in shifts
            }
        return {
            complex(shift): self._last_factorizations[complex(shift)].selected_inverse(
                self._density_extra_pattern
            )
            for shift in shifts
        }

    def _merge_charge_and_extra_entries(
        self,
        charge_entries: dict[complex, np.ndarray],
        extra_entries: dict[complex, np.ndarray],
        *,
        shifts: np.ndarray,
    ) -> dict[complex, np.ndarray]:
        merged: dict[complex, np.ndarray] = {}
        for shift in shifts:
            key = complex(shift)
            full_entries = np.zeros(
                self._density_pattern.pattern.nnz, dtype=np.complex128
            )
            if self._charge_to_density_source_positions.size:
                full_entries[self._charge_to_density_target_positions] = charge_entries[
                    key
                ][self._charge_to_density_source_positions]
            if self._extra_to_density_source_positions.size:
                full_entries[self._extra_to_density_target_positions] = extra_entries[
                    key
                ][self._extra_to_density_source_positions]
            merged[key] = full_entries
        return merged

    def density_values_from_charge_order(self, mu: float) -> np.ndarray:
        """Use only the factors from charge evaluation at this same mu."""
        if self._last_mu != float(mu) or self._last_terms is None:
            raise ValueError("Evaluate charge at the requested mu before density")
        terms = self._last_terms
        extra_entries = self._request_extra_density_entries_from_cached_factorizations(
            shifts=terms.shifts,
        )
        entries = self._merge_charge_and_extra_entries(
            self._last_charge_entries,
            extra_entries,
            shifts=terms.shifts,
        )
        return self._density_pattern.density_values_from_inverse_entries(
            entries,
            constant=terms.constant,
            shifts=terms.shifts,
            residues=terms.residues,
        )
