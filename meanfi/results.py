from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meanfi.errors import ErrorValues
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import _tb_type


@dataclass(frozen=True)
class DensityIntegrationInfo:
    """Internal statistics for a single density integration at fixed chemical potential."""

    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int
    error_estimate_available: bool
    num_threads: int | None = None
    charge: float | None = None
    charge_error: float | None = None

    requested_nk: int | None = None
    n_kpoints: int | None = None
    n_diagonalizations: int | None = None


@dataclass(frozen=True)
class FixedFillingInfo:
    """Internal statistics for a fixed-filling density calculation."""

    mu: float
    charge: float
    charge_error: float | None
    dcharge_dmu: float
    charge_evaluations: int
    charge_integration_calls: int
    density_integration_calls: int
    charge_n_kernel_evals: int
    density_n_kernel_evals: int
    n_kernel_evals: int
    unique_evals: int
    charge_n_evaluator_evals: int
    density_n_evaluator_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int
    charge_integral_atol: float
    density_atol: float
    density_rtol: float
    error_estimate_available: bool
    num_threads: int | None = None
    band_energy: float | None = None
    band_energy_integration_calls: int = 0
    band_energy_n_kernel_evals: int = 0

    requested_nk: int | None = None
    n_kpoints: int | None = None
    n_diagonalizations: int | None = None


@dataclass(frozen=True)
class AdaptiveSimplexInfo:
    """Mesh size and cumulative work for FermiSimplex integration."""

    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    refinements: int
    error_estimate_available: bool
    charge_evaluations: int | None = None
    charge_integration_calls: int | None = None
    density_integration_calls: int | None = None
    charge_error: float | None = None
    num_threads: int | None = None
    band_energy_integration_calls: int = 0
    band_energy_n_kernel_evals: int = 0

    requested_nk: int | None = None
    n_kpoints: int | None = None
    n_diagonalizations: int | None = None


@dataclass(frozen=True)
class PeriodicGridInfo:
    """Mesh size, retained storage, and cumulative work for periodic integration."""

    requested_nk: int | None
    n_kpoints: int
    grid_shape: tuple[int, ...]
    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_diagonalizations: int | None = None
    refinements: int = 0
    validation_evaluations: int = 0
    charge_evaluations: int = 0
    charge_integration_calls: int = 0
    density_integration_calls: int = 0
    charge_error: float | None = None
    error_estimate_available: bool = False
    spectrum_bytes: int = 0


@dataclass(frozen=True)
class DensityResult:
    """Density values tied to an explicit, possibly incomplete coordinate layout.

    Entries outside ``coordinates`` were not evaluated. Consequently a density
    can be converted to tight-binding matrix blocks only when every entry of each
    listed block is present.
    """

    coordinates: DensityCoordinates
    values: np.ndarray
    mu: float
    filling: float
    errors: ErrorValues
    statistics: AdaptiveSimplexInfo | PeriodicGridInfo | None = None

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=complex, copy=True)
        if values.ndim != 1:
            raise ValueError("density values must be one-dimensional")
        if values.size != self.coordinates.value_count:
            raise ValueError("density values do not match their coordinate layout")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def is_complete(self) -> bool:
        """Whether every matrix entry is present for every listed key."""

        return self.coordinates.is_full

    def covers(self, coordinates: DensityCoordinates) -> bool:
        """Whether all entries in ``coordinates`` are available in this result."""

        if self.coordinates.size != coordinates.size:
            return False
        return set(coordinates.entries) <= set(self.coordinates.entries)

    def values_for(self, coordinates: DensityCoordinates) -> np.ndarray:
        """Return values in another covered coordinate layout's order."""

        if self.coordinates.size != coordinates.size:
            raise ValueError("density coordinate matrix sizes do not match")
        indices = {entry: index for index, entry in enumerate(self.coordinates.entries)}
        missing = [entry for entry in coordinates.entries if entry not in indices]
        if missing:
            preview = ", ".join(map(str, missing[:3]))
            suffix = "" if len(missing) <= 3 else ", ..."
            raise ValueError(
                f"density is missing {len(missing)} required coordinate(s): "
                f"{preview}{suffix}"
            )
        return np.asarray(
            [self.values[indices[entry]] for entry in coordinates.entries],
            dtype=complex,
        )

    def select(self, coordinates: DensityCoordinates) -> DensityResult:
        """Return the same density restricted to a covered coordinate layout."""

        return DensityResult(
            coordinates=coordinates,
            values=self.values_for(coordinates),
            mu=self.mu,
            filling=self.filling,
            errors=self.errors,
            statistics=self.statistics,
        )

    def to_matrix(self) -> _tb_type:
        """Materialize complete tight-binding matrix blocks.

        Selected layouts deliberately cannot be materialized because filling
        uncomputed entries with zeros would change their physical meaning.
        """

        if not self.is_complete:
            raise ValueError(
                "cannot convert selected density coordinates to complete matrix "
                "blocks; request complete keys or use coordinates and values"
            )
        return self.coordinates.values_to_tb(self.values)

    @property
    def density_matrix(self) -> _tb_type:
        """Complete matrix blocks (compatibility alias for :meth:`to_matrix`)."""

        return self.to_matrix()


@dataclass(frozen=True)
class SCFIteration:
    """Physical and numerical values from one successful SCF evaluation."""

    step: int
    mu: float
    filling: float
    total_energy: float | None
    errors: ErrorValues


@dataclass(frozen=True)
class SCFResult:
    """A self-consistent mean-field state, or the last valid partial state."""

    density: DensityResult
    mean_field: dict[tuple[int, ...], Any]
    total_energy: float | None
    errors: ErrorValues
    history: tuple[SCFIteration, ...]
    converged: bool

    @property
    def mu(self) -> float:
        """Chemical potential of the final density evaluation."""

        return self.density.mu

    @property
    def filling(self) -> float:
        """Filling of the final density evaluation."""

        return self.density.filling
