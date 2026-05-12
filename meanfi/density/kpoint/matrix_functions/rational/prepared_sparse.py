from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import as_sparse, is_sparse_like

from ..base import RationalFOE
from ..common import spectral_interval, workspace_matrix, shift_by_mu
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
    _scheme_terms,
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
        self.matrix = as_sparse(workspace_matrix(matrix, self.workspace_dtype)).tocsr()
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
        self._density_coordinates = density_coordinates
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
        self._charge_cache: dict[
            float, tuple[float, int, complex, np.ndarray, np.ndarray]
        ] = {}
        self._sparse_terms_cache: dict[
            tuple[float, int, float], SparseRationalTerms
        ] = {}
        self._aaa_interval_cache: list[_AAAIntervalCacheEntry] = (
            shared_aaa_interval_cache if shared_aaa_interval_cache is not None else []
        )
        self._last_mu: float | None = None
        self._last_pole_count: int | None = None
        self._last_constant: complex = complex(0.0)
        self._last_shifts = np.empty(0, dtype=np.complex128)
        self._last_residues = np.empty(0, dtype=np.complex128)
        self._last_factorizations: dict[complex, SelectedInverseFactorization] = {}
        self._last_charge_entries: dict[complex, np.ndarray] = {}

    def _density_scalar_tolerance(self) -> float:
        return float(self.density_tolerance)

    def _charge_scalar_tolerance(self) -> float:
        weight_sum = float(np.sum(np.abs(self._charge_pattern.charge_weights)))
        if weight_sum <= 0.0:
            return self._density_scalar_tolerance()
        # Charge is a weighted trace, so the scalar Fermi-operator error must
        # shrink with the total trace weight to keep the filling solve stable.
        return max(
            np.finfo(float).eps,
            min(
                self._density_scalar_tolerance(),
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
            barycentric = _barycentric_evaluate(
                certification_grid,
                entry.support_x,
                entry.support_y,
                entry.weights,
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
        if scalar_tolerance is None:
            scalar_tolerance = self._density_scalar_tolerance()
        cache_key = (float(mu), int(pole_count), float(scalar_tolerance))
        cached = self._sparse_terms_cache.get(cache_key)
        if cached is not None:
            return cached

        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        lower, upper = spectral_interval(shifted)
        padding = 1e-12 * max(1.0, float(upper - lower))
        lower -= padding
        upper += padding
        if self.options.rational_scheme == "ozaki":
            constant, shifts, residues = _scheme_terms(
                self.options,
                pole_count,
                lower=lower,
                upper=upper,
                kT=self.kT,
            )
            terms = SparseRationalTerms(
                constant=constant,
                shifts=np.asarray(shifts, dtype=np.complex128),
                residues=np.asarray(residues, dtype=np.complex128),
                pole_count=int(pole_count),
            )
        else:
            cached_terms = self._aaa_cached_terms_for_interval(
                lower=lower,
                upper=upper,
                pole_cap=int(pole_count),
                scalar_tolerance=float(scalar_tolerance),
            )
            if cached_terms is not None:
                self._sparse_terms_cache[cache_key] = cached_terms
                return cached_terms
            terms, builder = _aaa_terms_for_interval(
                pole_cap=int(pole_count),
                lower=lower,
                upper=upper,
                kT=self.kT,
                initial_poles=int(self.options.initial_poles),
                scalar_tolerance=float(scalar_tolerance),
            )
            self._aaa_interval_cache.append(
                _AAAIntervalCacheEntry(
                    lower=lower,
                    upper=upper,
                    kT=self.kT,
                    support_x=np.asarray(builder.support_x, dtype=float),
                    support_y=np.asarray(builder.support_y, dtype=complex),
                    weights=np.asarray(builder.weights, dtype=complex),
                    terms=terms,
                )
            )
        self._sparse_terms_cache[cache_key] = terms
        return terms

    def _certified_sparse_terms(
        self,
        mu: float,
        *,
        scalar_tolerance: float | None = None,
    ) -> SparseRationalTerms:
        if scalar_tolerance is None:
            scalar_tolerance = self._density_scalar_tolerance()
        if self.options.rational_scheme == "aaa":
            try:
                return self._sparse_terms(
                    mu,
                    pole_count=int(self.options.max_poles),
                    scalar_tolerance=float(scalar_tolerance),
                )
            except ValueError as exc:
                if "AAA scalar certification failed" not in str(exc):
                    raise
                raise ValueError(
                    "Rational FOE did not converge within max_poles"
                ) from exc
        pole_count = int(self.options.initial_poles)
        while True:
            try:
                return self._sparse_terms(
                    mu,
                    pole_count=pole_count,
                    scalar_tolerance=float(scalar_tolerance),
                )
            except ValueError as exc:
                if pole_count == int(self.options.max_poles):
                    raise ValueError(
                        "Rational FOE did not converge within max_poles"
                    ) from exc
                pole_count = min(int(self.options.max_poles), 2 * pole_count)

    def _evaluate_charge_for_pole_count(
        self,
        mu: float,
        *,
        pole_count: int,
        terms: SparseRationalTerms | None = None,
        factorization_cache: dict[complex, SelectedInverseFactorization] | None = None,
    ) -> tuple[
        float,
        complex,
        np.ndarray,
        np.ndarray,
        dict[complex, SelectedInverseFactorization],
        dict[complex, np.ndarray],
    ]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        if terms is None:
            terms = self._sparse_terms(mu, pole_count=pole_count)
        constant = terms.constant
        shifts = terms.shifts
        residues = terms.residues
        active_factorizations = (
            {} if factorization_cache is None else dict(factorization_cache)
        )
        charge_entries: dict[complex, np.ndarray] = {}
        for shift in shifts:
            key = complex(shift)
            factorization = active_factorizations.get(key)
            if factorization is None:
                factorization = SelectedInverseFactorization()
                active_factorizations[key] = factorization
            factorization.factor(_sparse_shifted_matrix(shifted, key))
            charge_entries[key] = factorization.selected_inverse(
                self._charge_pattern.pattern
            )
        charge = self._charge_pattern.charge_from_inverse_entries(
            charge_entries,
            constant=constant,
            shifts=shifts,
            residues=residues,
        )
        return charge, constant, shifts, residues, active_factorizations, charge_entries

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, _pole_count, _constant, _shifts, _residues = self._charge_cache[mu]
            return charge, float("nan")

        if self.options.rational_scheme == "aaa":
            terms = self._certified_sparse_terms(
                mu,
                scalar_tolerance=self._charge_scalar_tolerance(),
            )
            (
                charge,
                _constant,
                _shifts,
                _residues,
                factorization_cache,
                charge_entries,
            ) = self._evaluate_charge_for_pole_count(
                mu,
                pole_count=terms.pole_count,
                terms=terms,
                factorization_cache=None,
            )
            self._charge_cache[mu] = (
                charge,
                terms.pole_count,
                terms.constant,
                np.asarray(terms.shifts, dtype=np.complex128),
                np.asarray(terms.residues, dtype=np.complex128),
            )
            self._last_mu = float(mu)
            self._last_pole_count = terms.pole_count
            self._last_constant = terms.constant
            self._last_shifts = np.asarray(terms.shifts, dtype=np.complex128)
            self._last_residues = np.asarray(terms.residues, dtype=np.complex128)
            self._last_factorizations = dict(factorization_cache)
            self._last_charge_entries = dict(charge_entries)
            return charge, float("nan")

        half_poles = int(self.options.initial_poles)
        (
            half_charge,
            _constant,
            _shifts,
            _residues,
            factorization_cache,
            _charge_entries,
        ) = self._evaluate_charge_for_pole_count(
            mu,
            pole_count=half_poles,
            factorization_cache=None,
        )

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            (
                full_charge,
                constant,
                shifts,
                residues,
                factorization_cache,
                charge_entries,
            ) = self._evaluate_charge_for_pole_count(
                mu,
                pole_count=pole_count,
                factorization_cache=factorization_cache,
            )
            if abs(full_charge - half_charge) <= self.charge_tolerance:
                self._charge_cache[mu] = (
                    full_charge,
                    pole_count,
                    constant,
                    np.asarray(shifts, dtype=np.complex128),
                    np.asarray(residues, dtype=np.complex128),
                )
                self._last_mu = float(mu)
                self._last_pole_count = pole_count
                self._last_constant = constant
                self._last_shifts = np.asarray(shifts, dtype=np.complex128)
                self._last_residues = np.asarray(residues, dtype=np.complex128)
                self._last_factorizations = dict(factorization_cache)
                self._last_charge_entries = dict(charge_entries)
                return full_charge, float("nan")

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_charge = full_charge

    def _density_values_from_inverse_entries(
        self,
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
        inverse_entries: dict[complex, np.ndarray],
    ) -> np.ndarray:
        return self._density_pattern.density_values_from_inverse_entries(
            inverse_entries,
            constant=constant,
            shifts=shifts,
            residues=residues,
        )

    def _request_density_inverse_entries(
        self,
        mu: float,
        *,
        pole_count: int,
        factorization_cache: dict[complex, SelectedInverseFactorization] | None = None,
        terms: SparseRationalTerms | None = None,
    ) -> tuple[
        complex,
        np.ndarray,
        np.ndarray,
        dict[complex, SelectedInverseFactorization],
        dict[complex, np.ndarray],
    ]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        resolved_terms = (
            terms
            if terms is not None
            else self._sparse_terms(mu, pole_count=pole_count)
        )
        constant = resolved_terms.constant
        shifts = resolved_terms.shifts
        residues = resolved_terms.residues
        factorizations = (
            {} if factorization_cache is None else dict(factorization_cache)
        )
        inverse_entries: dict[complex, np.ndarray] = {}
        for shift in shifts:
            key = complex(shift)
            factorization = factorizations.get(key)
            if factorization is None:
                factorization = SelectedInverseFactorization()
                factorizations[key] = factorization
            factorization.factor(_sparse_shifted_matrix(shifted, key))
            inverse_entries[key] = factorization.selected_inverse(
                self._density_pattern.pattern
            )
        return constant, shifts, residues, factorizations, inverse_entries

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

    def density_values(self, mu: float, *, tolerance: float) -> np.ndarray:
        if self.options.rational_scheme == "aaa":
            terms = self._certified_sparse_terms(mu)
            constant, shifts, residues, factorizations, inverse_entries = (
                self._request_density_inverse_entries(
                    mu,
                    pole_count=terms.pole_count,
                    factorization_cache=None,
                    terms=terms,
                )
            )
            self._last_mu = float(mu)
            self._last_pole_count = terms.pole_count
            self._last_constant = terms.constant
            self._last_shifts = np.asarray(terms.shifts, dtype=np.complex128)
            self._last_residues = np.asarray(terms.residues, dtype=np.complex128)
            self._last_factorizations = dict(factorizations)
            self._last_charge_entries = {}
            return self._density_values_from_inverse_entries(
                constant=constant,
                shifts=shifts,
                residues=residues,
                inverse_entries=inverse_entries,
            )

        half_poles = int(self.options.initial_poles)
        half_constant, half_shifts, half_residues, factorization_cache, half_entries = (
            self._request_density_inverse_entries(
                mu,
                pole_count=half_poles,
                factorization_cache=None,
            )
        )
        half_values = self._density_values_from_inverse_entries(
            constant=half_constant,
            shifts=half_shifts,
            residues=half_residues,
            inverse_entries=half_entries,
        )

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            (
                full_constant,
                full_shifts,
                full_residues,
                factorization_cache,
                full_entries,
            ) = self._request_density_inverse_entries(
                mu,
                pole_count=pole_count,
                factorization_cache=factorization_cache,
            )
            full_values = self._density_values_from_inverse_entries(
                constant=full_constant,
                shifts=full_shifts,
                residues=full_residues,
                inverse_entries=full_entries,
            )
            if (
                float(np.max(np.abs(full_values - half_values), initial=0.0))
                <= tolerance
            ):
                self._last_mu = float(mu)
                self._last_pole_count = pole_count
                self._last_constant = full_constant
                self._last_shifts = np.asarray(full_shifts, dtype=np.complex128)
                self._last_residues = np.asarray(full_residues, dtype=np.complex128)
                self._last_factorizations = dict(factorization_cache)
                self._last_charge_entries = {}
                return full_values

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_values = full_values

    def density_values_from_charge_order(
        self, mu: float, basis: np.ndarray | None = None
    ) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError(
                "Charge pole count is unavailable for the requested chemical potential"
            )
        pole_count = charge_cache[1]
        cached_constant = charge_cache[2]
        cached_shifts = np.asarray(charge_cache[3], dtype=np.complex128)
        cached_residues = np.asarray(charge_cache[4], dtype=np.complex128)
        if (
            self._last_mu == float(mu)
            and self._last_pole_count == pole_count
            and self._last_charge_entries
        ):
            extra_entries = (
                self._request_extra_density_entries_from_cached_factorizations(
                    shifts=self._last_shifts,
                )
            )
            inverse_entries = self._merge_charge_and_extra_entries(
                self._last_charge_entries,
                extra_entries,
                shifts=self._last_shifts,
            )
            return self._density_values_from_inverse_entries(
                constant=self._last_constant,
                shifts=self._last_shifts,
                residues=self._last_residues,
                inverse_entries=inverse_entries,
            )
        constant, shifts, residues, factorizations, inverse_entries = (
            self._request_density_inverse_entries(
                mu,
                pole_count=pole_count,
                terms=SparseRationalTerms(
                    constant=cached_constant,
                    shifts=cached_shifts,
                    residues=cached_residues,
                    pole_count=pole_count,
                ),
            )
        )
        self._last_mu = float(mu)
        self._last_pole_count = pole_count
        self._last_constant = constant
        self._last_shifts = np.asarray(shifts, dtype=np.complex128)
        self._last_residues = np.asarray(residues, dtype=np.complex128)
        self._last_factorizations = dict(factorizations)
        self._last_charge_entries = {}
        return self._density_values_from_inverse_entries(
            constant=constant,
            shifts=shifts,
            residues=residues,
            inverse_entries=inverse_entries,
        )
