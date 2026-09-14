from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from meanfi.errors import ErrorValues
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import _tb_type


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
    charge: float | None = None
    band_energy: float | None = None


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


def _readonly_vector(values, *, dtype, name: str) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DensityEntries:
    """Immutable computed entries and their optional integration errors.

    Entries outside ``coordinates`` are unknown, so selected layouts cannot be
    materialized as complete matrix blocks. Result metadata can share this
    payload without copying its arrays.
    """

    coordinates: DensityCoordinates
    values: np.ndarray
    errors: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = _readonly_vector(self.values, dtype=complex, name="density values")
        if values.size != self.coordinates.value_count:
            raise ValueError("density values do not match their coordinate layout")
        object.__setattr__(self, "values", values)
        if self.errors is not None:
            errors = _readonly_vector(self.errors, dtype=float, name="density errors")
            if errors.size != values.size:
                raise ValueError("density errors do not match their coordinate layout")
            if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
                raise ValueError("density errors must be finite and non-negative")
            object.__setattr__(self, "errors", errors)

    def to_tb(self, *, sparse: bool = False) -> _tb_type:
        """Materialize complete blocks, rejecting uncomputed matrix entries."""
        if not self.coordinates.is_full:
            raise ValueError(
                "cannot convert selected density coordinates to complete matrix "
                "blocks; request complete keys or use coordinates and values"
            )
        return self.coordinates.values_to_tb(self.values, sparse=sparse)


@dataclass(frozen=True)
class DensityResult:
    """Density entries with physical values, errors, and integration statistics."""

    entries: DensityEntries
    mu: float
    filling: float
    errors: ErrorValues
    statistics: AdaptiveSimplexInfo | PeriodicGridInfo | None = None
    band_energy: float | None = None

    @property
    def coordinates(self) -> DensityCoordinates:
        return self.entries.coordinates

    @property
    def values(self) -> np.ndarray:
        return self.entries.values

    @property
    def entry_errors(self) -> np.ndarray | None:
        """Per-entry integration errors, or None when not estimated."""
        return self.entries.errors

    @property
    def is_complete(self) -> bool:
        """Whether every matrix entry is present for every listed key."""
        return self.coordinates.is_full

    def covers(self, coordinates: DensityCoordinates) -> bool:
        """Whether all entries in ``coordinates`` are available in this result."""
        return self.coordinates.size == coordinates.size and set(
            coordinates.entries
        ) <= set(self.coordinates.entries)

    def _indices_for(self, coordinates: DensityCoordinates) -> list[int]:
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
        return [indices[entry] for entry in coordinates.entries]

    def values_for(self, coordinates: DensityCoordinates) -> np.ndarray:
        """Return values in another covered coordinate layout's order."""
        if coordinates is self.coordinates:
            return self.values
        return self.values[self._indices_for(coordinates)]

    def select(self, coordinates: DensityCoordinates) -> DensityResult:
        """Restrict entries and entry errors, preserving evaluation metadata."""
        if coordinates is self.coordinates:
            return self
        indices = self._indices_for(coordinates)
        return replace(
            self,
            entries=DensityEntries(
                coordinates,
                self.values[indices],
                None if self.entry_errors is None else self.entry_errors[indices],
            ),
        )

    def to_tb(self, *, sparse: bool = False) -> _tb_type:
        """Materialize complete blocks; selected layouts contain unknown entries."""
        return self.entries.to_tb(sparse=sparse)


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
    mean_field: _tb_type
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
