"""Private coordinate-valued density payloads.

Selected density entries are not matrices: entries outside their coordinate
layout were not evaluated and must never be represented as physical zeros at a
public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.errors import ErrorValues
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import _tb_type


def _readonly_vector(values, *, dtype, name: str) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DensitySlice:
    """Values for an explicit, possibly incomplete set of density entries."""

    coordinates: DensityCoordinates
    values: np.ndarray
    errors: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = _readonly_vector(self.values, dtype=complex, name="density values")
        if values.size != self.coordinates.value_count:
            raise ValueError("density values do not match their coordinate layout")
        object.__setattr__(self, "values", values)

        if self.errors is None:
            return
        errors = _readonly_vector(self.errors, dtype=float, name="density errors")
        if errors.size != values.size:
            raise ValueError("density errors do not match their coordinate layout")
        if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
            raise ValueError("density errors must be finite and non-negative")
        object.__setattr__(self, "errors", errors)

    @classmethod
    def from_tb(
        cls,
        coordinates: DensityCoordinates,
        density_matrix: _tb_type,
        density_matrix_error: _tb_type | None,
    ) -> DensitySlice:
        errors = (
            None
            if density_matrix_error is None
            else np.real(coordinates.values_from_tb(density_matrix_error))
        )
        return cls(
            coordinates=coordinates,
            values=coordinates.values_from_tb(density_matrix),
            errors=errors,
        )

    def select_keys(self, keys: list[tuple[int, ...]]) -> DensitySlice:
        requested = [tuple(key) for key in keys]
        positions: list[int] = []
        entries = self.coordinates.entries
        for key in requested:
            positions.extend(
                index for index, entry in enumerate(entries) if entry[0] == key
            )
        selected_entries = tuple(entries[index] for index in positions)
        coordinates = DensityCoordinates.from_entries(
            size=self.coordinates.size,
            keys=requested,
            entries=selected_entries,
            allow_empty=True,
        )
        if coordinates is None:  # pragma: no cover - allow_empty guarantees this
            raise RuntimeError("selected density coordinates unexpectedly missing")
        indices = np.asarray(positions, dtype=int)
        return DensitySlice(
            coordinates=coordinates,
            values=self.values[indices],
            errors=None if self.errors is None else self.errors[indices],
        )

    def to_full_tb(self) -> _tb_type:
        """Assemble a public density matrix only from a complete layout."""

        if not self.coordinates.is_full:
            raise ValueError(
                "cannot expose selected density entries as a complete density matrix"
            )
        return self.coordinates.values_to_tb(self.values)


@dataclass(frozen=True)
class DensityEvaluation:
    """Private result of one density integration."""

    density: DensitySlice
    mu: float
    filling: float
    errors: ErrorValues
    integration: object
    statistics: object
    band_energy: float | None = None
