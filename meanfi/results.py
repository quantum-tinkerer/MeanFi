from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Callable

import numpy as np

from meanfi.errors import ErrorValues
from meanfi.space.coordinates import DensityCoordinates, _assemble_blocks
from meanfi.tb.ops import _tb_type


@dataclass(frozen=True, kw_only=True)
class IntegrationInfo:
    """Common mesh and work counts, with optional method-specific details.

    Refinements count local simplex subdivisions or global grid doublings.
    error_estimate_available refers to momentum integration only.
    n_energy_evaluations counts new centroid eigenvalue evaluations, included
    in n_kernel_evals and n_diagonalizations. These are not retained by the
    native mesh cache. n_energy_simplices counts the evaluated energy partition,
    including previews; n_leaves remains the active mesh count.
    """

    n_kpoints: int
    n_kernel_evals: int
    requested_nk: int | None = None
    n_diagonalizations: int = 0
    refinements: int = 0
    p_refinements: int = 0
    charge_evaluations: int = 0
    charge_integration_calls: int = 0
    density_integration_calls: int = 0
    error_estimate_available: bool = False
    grid_shape: tuple[int, ...] | None = None
    n_cached_nodes: int | None = None
    n_leaves: int | None = None
    num_threads: int | None = None
    spectrum_bytes: int = 0
    n_energy_evaluations: int = 0
    n_energy_simplices: int = 0


def _readonly_vector(values, *, dtype, name: str) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class _DensityEntries:
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
            if np.any(errors < 0.0):
                raise ValueError("density errors must be finite and non-negative")
            object.__setattr__(self, "errors", errors)

    def to_tb(self, *, sparse: bool = False) -> _tb_type:
        """Materialize complete blocks, rejecting uncomputed matrix entries."""
        if not self.coordinates.is_full:
            raise ValueError(
                "cannot convert selected density coordinates to complete matrix "
                "blocks; request complete keys or use coordinates and values"
            )
        return _assemble_blocks(self.coordinates, self.values, sparse=sparse)

    def trace(self, size: int | None = None) -> float | None:
        """Local trace, or None when any required diagonal entry is missing."""
        size = self.coordinates.size if size is None else size
        for key, rows, cols, value_slice in self.coordinates.iter_key_coordinates():
            if any(key):
                continue
            diagonal = (rows == cols) & (rows < size)
            if np.count_nonzero(diagonal) == size:
                return float(self.values[value_slice][diagonal].real.sum())
        return None


@dataclass(frozen=True)
class DensityResult:
    """Density entries and metadata of the complete evaluated state.

    Fixed-filling results retain the charge-stage filling and root residual.
    At fixed mu, filling is derived from available density data or is None.
    The independent charge-integration estimate is unavailable at fixed mu.
    ``entropy`` and ``band_energy`` are per cell per physical orbital; entropy
    is in units of Boltzmann's constant. The band energy belongs to the input
    quadratic Hamiltonian (with BdG normal ordering),
    before correcting for interaction double counting. FermiSimplex uses a
    vertex/centroid spectral-hinge rule plus mu times the integrated density
    trace. This does not change the stored density.
    Model-based results retain known ``internal_energy``; ``free_energy`` subtracts
    ``kT * entropy``.
    ``density_filling`` is particle count on the density partition when known;
    ``filling`` can instead report the preceding charge-root result.
    Free energy is also None when entropy was not requested.
    Both are None when the model energy cannot be determined. Selection preserves
    these scalars; unknown real-space entries are never filled to obtain them.
    """

    entries: _DensityEntries
    mu: float
    filling: float | None
    errors: ErrorValues
    statistics: IntegrationInfo | None = None
    band_energy: float | None = None
    entropy: float | None = None
    kT: float = 0.0
    internal_energy: float | None = None
    density_filling: float | None = None
    _energy_evaluation: Callable[[], DensityResult] | None = field(
        default=None, repr=False, compare=False
    )

    def _with_energy(self) -> DensityResult:
        """Finish a deferred observable on its original mesh, then release it."""
        if self._energy_evaluation is None:
            return self
        return self._energy_evaluation().select(self.coordinates)

    @property
    def free_energy(self) -> float | None:
        """Model energy minus kT * entropy, when the model energy is known."""
        if self.internal_energy is None or self.entropy is None:
            return None
        return self.internal_energy - self.kT * self.entropy

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
            entries=_DensityEntries(
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
    internal_energy: float | None
    errors: ErrorValues


@dataclass(frozen=True)
class SCFResult:
    """An evaluated Hamiltonian/density pair, converged or the last valid state.

    ``mean_field`` is the input correction that produced ``density``. The next
    correction is ``model.mean_field(result.density)``; on nonconvergence it may
    differ substantially from the input correction.
    """

    density: DensityResult
    mean_field: _tb_type
    history: tuple[SCFIteration, ...]
    converged: bool

    @property
    def internal_energy(self) -> float | None:
        return self.density.internal_energy

    @property
    def free_energy(self) -> float | None:
        return self.density.free_energy

    @property
    def errors(self) -> ErrorValues:
        return self.density.errors

    @property
    def entropy(self) -> float | None:
        """Entropy per cell per physical orbital, in units of Boltzmann's constant."""
        return self.density.entropy

    @property
    def mu(self) -> float:
        """Chemical potential of the final density evaluation."""

        return self.density.mu

    @property
    def filling(self) -> float | None:
        """Filling of the final density evaluation."""

        return self.density.filling
