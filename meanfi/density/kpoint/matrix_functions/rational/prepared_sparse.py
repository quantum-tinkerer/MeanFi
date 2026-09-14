from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.errors import ConvergenceError

from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import as_sparse, is_sparse_like

from ..base import RationalFOE
from ..common import spectral_interval, shift_by_mu
from ..mumps_backend import (
    SelectedInverseFactorization,
    build_selected_inverse_pattern,
)
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
    thermal_errors,
    thermal_targets,
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
        thermodynamic_tolerance: float | None = None,
        workspace_dtype: np.dtype = np.dtype(complex),
        trace_weights_diag: np.ndarray | None = None,
        shared_aaa_interval_cache: list[_AAAIntervalCacheEntry] | None = None,
    ) -> None:
        if not is_sparse_like(matrix):
            raise ValueError("Sparse MUMPS-backed RationalFOE requires sparse matrices")

        self.workspace_dtype = np.dtype(workspace_dtype)
        self.matrix = as_sparse(matrix).astype(self.workspace_dtype).tocsr()
        self.kT = float(kT)
        self.q_diag = np.asarray(q_diag, dtype=float)
        self.options = options
        self.charge_tolerance = float(charge_tolerance)
        self.density_tolerance = float(density_tolerance)
        self.thermodynamic_tolerance = thermodynamic_tolerance
        self.size = int(getattr(matrix, "shape")[0])
        self._trace_weights = (
            np.ones(self.size, dtype=float)
            if trace_weights_diag is None
            else np.asarray(trace_weights_diag, dtype=float)
        )
        self._charge_pattern = build_sparse_charge_pattern(
            self._trace_weights, include_all=thermodynamic_tolerance is not None
        )
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

    def _scalar_tolerances(self, lower: float, upper: float, mu: float) -> np.ndarray:
        density_tolerance = self._charge_scalar_tolerance()
        if self.thermodynamic_tolerance is None:
            return np.array([density_tolerance])
        trace_tolerance = self.thermodynamic_tolerance / self.size
        return np.array(
            [
                min(
                    density_tolerance,
                    # Bound the unshifted Hamiltonian used for band energy.
                    trace_tolerance
                    / max(
                        1.0,
                        max(abs(lower), abs(upper))
                        + abs(mu) * np.max(np.abs(self.q_diag)),
                    ),
                ),
                trace_tolerance,
            ]
        )

    def _sparse_terms(self, mu: float) -> SparseRationalTerms:
        pole_count = self.options.max_poles
        shifted = shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype)
        lower, upper = spectral_interval(shifted)
        padding = 1e-12 * max(1.0, float(upper - lower))
        lower, upper = lower - padding, upper + padding
        tolerances = self._scalar_tolerances(lower, upper, mu)
        grid = _aaa_sample_grid(
            lower, upper, kT=self.kT, count=max(2048, 32 * pole_count)
        )
        targets = thermal_targets(grid, self.kT)[:, : tolerances.size]
        for entry in self._aaa_interval_cache:
            if (
                entry.kT != self.kT
                or lower < entry.lower
                or upper > entry.upper
                or entry.terms.pole_count > pole_count
                or (tolerances.size == 2 and entry.terms.entropy_residues is None)
            ):
                continue
            if np.all(thermal_errors(entry.terms, grid, targets) <= tolerances):
                return entry.terms
        try:
            terms = _aaa_terms_for_interval(
                pole_cap=pole_count,
                lower=lower,
                upper=upper,
                kT=self.kT,
                initial_poles=self.options.initial_poles,
                scalar_tolerance=tolerances[0],
                entropy_tolerance=tolerances[1] if tolerances.size == 2 else None,
            )
        except ValueError as exc:
            raise ConvergenceError(str(exc)) from exc
        # Keep one scalar fit shared across k-points, never their factorizations.
        self._aaa_interval_cache[:] = [
            _AAAIntervalCacheEntry(lower, upper, self.kT, terms)
        ]
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
            entries[key] = factorization.selected_inverse(self._charge_pattern.pattern)
        charge = self._charge_pattern.charge_from_inverse_entries(
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
        if terms.entropy_residues is None:
            raise ValueError("Prepare the node with thermodynamic_tolerance first")
        energy = float(np.real(terms.constant * np.sum(self.matrix.diagonal())))
        entropy = float(np.real(terms.entropy_constant * self.size))
        for shift, residue, entropy_residue in zip(
            terms.shifts, terms.residues, terms.entropy_residues, strict=True
        ):
            diagonal = self._last_charge_entries[complex(shift)][
                self._charge_pattern.diagonal_positions
            ]
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
