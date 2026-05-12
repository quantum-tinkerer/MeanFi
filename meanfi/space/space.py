from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from meanfi.space.coordinates import DensityCoordinates, DensityEntry
from meanfi.space.reducers import (
    LinearConstraintReducer,
    OrbitReducer,
    complex_to_real,
    real_to_complex,
)
from meanfi.space.selection import select_required_coordinates
from meanfi.space.support import (
    ActiveCoordinateSupport,
    bdg_active_support,
    normal_active_support,
)
from meanfi.space.symmetry import HermiticityConstraint, ParticleHoleConstraint
from meanfi.tb.ops import _tb_type, is_sparse_like

if TYPE_CHECKING:
    from meanfi.model import Model


@dataclass(frozen=True)
class ActiveSCFSpace:
    """Minimal real variables used by one mean-field SCF problem."""

    active_coordinates: DensityCoordinates
    required_coordinates: DensityCoordinates
    basis: np.ndarray
    required_real_rows: np.ndarray
    interaction_keys: list[tuple[int, ...]]
    density_keys: list[tuple[int, ...]]
    onsite: tuple[int, ...]

    @property
    def active_entries(self) -> tuple[DensityEntry, ...]:
        return self.active_coordinates.entries

    @property
    def num_params(self) -> int:
        return int(self.basis.shape[1])

    @classmethod
    def normal(cls, model: Model) -> ActiveSCFSpace:
        support = normal_active_support(model)
        entries = support.coordinates.entries
        basis = OrbitReducer(entries).basis((HermiticityConstraint(),))
        basis = LinearConstraintReducer(
            entries,
            ndof=model._ndof,
            family="normal",
        ).basis(basis, getattr(model, "spatial_symmetries", ()))
        return cls._from_support(support, basis)

    @classmethod
    def bdg(cls, model: Model) -> ActiveSCFSpace:
        support = bdg_active_support(model)
        entries = support.coordinates.entries
        basis = OrbitReducer(entries).basis(
            (
                HermiticityConstraint(electron_ndof=model._ndof),
                ParticleHoleConstraint(model._ndof),
            )
        )
        basis = LinearConstraintReducer(
            entries,
            ndof=model._ndof,
            family="bdg",
        ).basis(basis, getattr(model, "spatial_symmetries", ()))
        return cls._from_support(support, basis)

    @classmethod
    def from_model(cls, model: Model) -> ActiveSCFSpace:
        return cls.bdg(model) if model.superconducting else cls.normal(model)

    @classmethod
    def _from_support(
        cls,
        support: ActiveCoordinateSupport,
        basis: np.ndarray,
    ) -> ActiveSCFSpace:
        selected = select_required_coordinates(support.coordinates, basis)
        return cls(
            active_coordinates=support.coordinates,
            required_coordinates=selected.coordinates,
            basis=np.asarray(basis, dtype=float),
            required_real_rows=selected.real_rows,
            interaction_keys=support.interaction_keys,
            density_keys=support.density_keys,
            onsite=support.onsite,
        )

    def required_realspace_entries(self) -> tuple[DensityEntry, ...]:
        return self.required_coordinates.entries

    def required_density_coordinates_for(self, tb: _tb_type) -> DensityCoordinates | None:
        if self.required_coordinates.value_count == 0:
            return None
        if any(is_sparse_like(matrix) for matrix in tb.values()):
            return self.required_coordinates
        return None

    def params_from_required_entries(self, values: np.ndarray) -> np.ndarray:
        real_values = complex_to_real(values)
        if real_values.size != self.required_real_rows.size:
            raise ValueError("values do not match required real-space entries")
        if self.num_params == 0:
            return np.empty(0, dtype=float)
        sample_basis = self.basis[self.required_real_rows, :]
        params, *_ = np.linalg.lstsq(sample_basis, real_values, rcond=None)
        return np.asarray(params, dtype=float)

    def params_from_meanfield_input(self, rho: _tb_type) -> np.ndarray:
        return self.params_from_required_entries(
            self.required_coordinates.values_from_tb(rho)
        )

    def meanfield_input_from_params(self, params: np.ndarray) -> _tb_type:
        params = np.asarray(params, dtype=float).reshape(-1)
        if params.size != self.num_params:
            raise ValueError("params has the wrong length for this active SCF space")
        real_values = self.basis @ params
        return self.active_coordinates.values_to_tb(real_to_complex(real_values))

    def project_meanfield_input(self, rho: _tb_type) -> _tb_type:
        return self.meanfield_input_from_params(self.params_from_meanfield_input(rho))
