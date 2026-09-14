from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from numbers import Integral

import numpy as np

from meanfi.space.coordinates import DensityEntry, opposite_key


@dataclass(frozen=True)
class SpatialSymmetry:
    """Unitary or antiunitary spatial symmetry acting on density entries."""

    lattice_matrix: np.ndarray
    unitaries_by_shift: dict[tuple[int, ...], np.ndarray]
    antiunitary: bool = False

    def __post_init__(self) -> None:
        lattice_matrix = np.array(self.lattice_matrix, copy=True)
        if not np.all(np.isfinite(lattice_matrix)) or not np.all(
            lattice_matrix == np.round(lattice_matrix)
        ):
            raise ValueError(
                "SpatialSymmetry.lattice_matrix must contain finite integers"
            )
        lattice_matrix = lattice_matrix.astype(int)
        if (
            lattice_matrix.ndim != 2
            or lattice_matrix.shape[0] != lattice_matrix.shape[1]
        ):
            raise ValueError("SpatialSymmetry.lattice_matrix must be square")
        lattice_matrix.setflags(write=False)
        object.__setattr__(self, "lattice_matrix", lattice_matrix)
        if any(
            isinstance(component, bool) or not isinstance(component, Integral)
            for shift in self.unitaries_by_shift
            for component in shift
        ):
            raise ValueError("SpatialSymmetry shifts must contain integers")
        normalized = {
            tuple(int(component) for component in shift): np.array(
                unitary, dtype=complex, copy=True
            )
            for shift, unitary in self.unitaries_by_shift.items()
        }
        if not normalized:
            raise ValueError(
                "SpatialSymmetry requires at least one shift/unitary block"
            )
        ndim = lattice_matrix.shape[0]
        for shift, unitary in normalized.items():
            if len(shift) != ndim:
                raise ValueError("SpatialSymmetry shifts must match lattice dimension")
            if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
                raise ValueError("SpatialSymmetry unitary blocks must be square")
            if not np.all(np.isfinite(unitary)):
                raise ValueError("SpatialSymmetry blocks must contain finite values")
            unitary.setflags(write=False)
        object.__setattr__(self, "unitaries_by_shift", MappingProxyType(normalized))


@dataclass(frozen=True)
class HermiticityConstraint:
    """Constraint ``rho_ab(R) = conj(rho_ba(-R))``."""

    electron_ndof: int | None = None

    def partner(self, entry: DensityEntry) -> DensityEntry | None:
        key, row, col = entry
        if self.electron_ndof is not None and (
            row >= self.electron_ndof or col >= self.electron_ndof
        ):
            return None
        return opposite_key(key), col, row


@dataclass(frozen=True)
class ParticleHoleConstraint:
    """Anomalous BdG constraint ``F_ab(R) = -F_ba(-R)``."""

    ndof: int

    def partner(self, entry: DensityEntry) -> DensityEntry | None:
        key, row, col = entry
        if row >= self.ndof or col < self.ndof:
            return None
        return opposite_key(key), col - self.ndof, self.ndof + row
