from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from meanfi.space.coordinates import DensityCoordinates, DensityEntry
from meanfi.space.reducers import (
    LinearConstraintReducer,
    OrbitReducer,
    _warn_missing_partner,
    complex_to_real,
    real_to_complex,
)
from meanfi.space.selection import select_required_coordinates
from meanfi.space.support import (
    bdg_active_support,
    normal_active_support,
)
from meanfi.space.symmetry import HermiticityConstraint, ParticleHoleConstraint
from meanfi.tb.ops import _tb_type
from meanfi.tb.storage import prefers_sparse_storage

if TYPE_CHECKING:
    from meanfi.model import Model


_DENSE_BASIS_LIMIT_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class _OrbitParametrization:
    required_coordinates: DensityCoordinates
    required_value_rows: np.ndarray
    active_real_param: np.ndarray
    active_real_sign: np.ndarray
    active_imag_param: np.ndarray
    active_imag_sign: np.ndarray

    @property
    def num_params(self) -> int:
        return self.required_value_rows.size

    def params_from_real_values(self, values: np.ndarray) -> np.ndarray:
        return values[self.required_value_rows]

    def values_from_params(self, params: np.ndarray) -> np.ndarray:
        values = np.zeros(self.active_real_param.size, dtype=complex)
        real_mask = self.active_real_param >= 0
        values[real_mask] = (
            self.active_real_sign[real_mask] * params[self.active_real_param[real_mask]]
        )
        imag_mask = self.active_imag_param >= 0
        values[imag_mask] += (
            1j
            * self.active_imag_sign[imag_mask]
            * params[self.active_imag_param[imag_mask]]
        )
        return values


@dataclass(frozen=True)
class _DenseParametrization:
    required_coordinates: DensityCoordinates
    required_value_rows: np.ndarray
    basis: np.ndarray
    required_to_params: np.ndarray

    @property
    def num_params(self) -> int:
        return self.basis.shape[1]

    def params_from_real_values(self, values: np.ndarray) -> np.ndarray:
        return self.required_to_params @ values[self.required_value_rows]

    def values_from_params(self, params: np.ndarray) -> np.ndarray:
        return real_to_complex(self.basis @ params)


@dataclass(frozen=True)
class ActiveSCFSpace:
    """Minimal real variables and their density reconstruction for one SCF problem."""

    active_coordinates: DensityCoordinates
    parametrization: _OrbitParametrization | _DenseParametrization
    interaction_keys: list[tuple[int, ...]]
    density_keys: list[tuple[int, ...]]
    onsite: tuple[int, ...]
    sparse: bool

    @property
    def required_coordinates(self) -> DensityCoordinates:
        return self.parametrization.required_coordinates

    @property
    def num_params(self) -> int:
        return self.parametrization.num_params

    @classmethod
    def from_model(cls, model: Model) -> ActiveSCFSpace:
        if model.superconducting:
            support = bdg_active_support(model)
            family = "bdg"
            constraints = (
                HermiticityConstraint(electron_ndof=model._ndof),
                ParticleHoleConstraint(model._ndof),
            )
        else:
            support = normal_active_support(model)
            family = "normal"
            constraints = (HermiticityConstraint(),)

        if model.spatial_symmetries:
            entries = support.coordinates.entries
            _raise_if_dense_basis_too_large(len(entries), family=family)
            basis = OrbitReducer(entries).basis(constraints)
            basis = LinearConstraintReducer(
                entries, ndof=model._ndof, family=family
            ).basis(basis, model.spatial_symmetries)
            selected = select_required_coordinates(support.coordinates, basis)
            sample_basis = basis[selected.active_real_rows, :]
            parametrization = _DenseParametrization(
                required_coordinates=selected.coordinates,
                required_value_rows=selected.value_real_rows,
                basis=basis,
                required_to_params=np.linalg.inv(sample_basis),
            )
        else:
            parametrization = _orbit_parametrization(
                support.coordinates, constraints=constraints
            )

        return cls(
            active_coordinates=support.coordinates,
            parametrization=parametrization,
            interaction_keys=support.interaction_keys,
            density_keys=support.density_keys,
            onsite=support.onsite,
            sparse=prefers_sparse_storage(model.h_0, model.h_int),
        )

    def params_from_required_entries(self, values: np.ndarray) -> np.ndarray:
        real_values = complex_to_real(values)
        if real_values.size != 2 * self.required_coordinates.value_count:
            raise ValueError("values do not match required real-space entries")
        return self.parametrization.params_from_real_values(real_values)

    def params_from_meanfield_input(self, rho: _tb_type) -> np.ndarray:
        return self.params_from_required_entries(
            self.required_coordinates.values_from_tb(rho)
        )

    def meanfield_input_from_params(self, params: np.ndarray) -> _tb_type:
        params = np.asarray(params, dtype=float).reshape(-1)
        if params.size != self.num_params:
            raise ValueError("params has the wrong length for this active SCF space")
        return self.active_coordinates.values_to_tb(
            self.parametrization.values_from_params(params), sparse=self.sparse
        )

    def project_meanfield_input(self, rho: _tb_type) -> _tb_type:
        return self.meanfield_input_from_params(self.params_from_meanfield_input(rho))


def _orbit_parametrization(
    active_coordinates: DensityCoordinates,
    *,
    constraints: tuple[HermiticityConstraint | ParticleHoleConstraint, ...],
) -> _OrbitParametrization:
    entries = active_coordinates.entries
    value_count = len(entries)
    index = {entry: position for position, entry in enumerate(entries)}

    active_real_param = np.full(value_count, -1, dtype=int)
    active_imag_param = np.full(value_count, -1, dtype=int)
    active_real_sign = np.zeros(value_count, dtype=float)
    active_imag_sign = np.zeros(value_count, dtype=float)
    visited = np.zeros(value_count, dtype=bool)

    required_entries: list[DensityEntry] = []
    required_param_rows: list[tuple[int, bool]] = []
    parameter_count = 0

    def add_required(position: int) -> int:
        required_position = len(required_entries)
        required_entries.append(entries[position])
        return required_position

    def add_real_param(position: int, *, sign: float, required_position: int) -> None:
        nonlocal parameter_count
        active_real_param[position] = parameter_count
        active_real_sign[position] = sign
        required_param_rows.append((required_position, False))
        parameter_count += 1

    def add_imag_param(position: int, *, sign: float, required_position: int) -> None:
        nonlocal parameter_count
        active_imag_param[position] = parameter_count
        active_imag_sign[position] = sign
        required_param_rows.append((required_position, True))
        parameter_count += 1

    for position, entry in enumerate(entries):
        if visited[position]:
            continue
        partners = _constraint_partners(entry, constraints)
        if not partners:
            visited[position] = True
            required_position = add_required(position)
            add_real_param(position, sign=1.0, required_position=required_position)
            add_imag_param(position, sign=1.0, required_position=required_position)
            continue
        if len(partners) > 1:
            raise NotImplementedError(
                "compact SCF orbit parametrization supports one pair constraint "
                "per active entry"
            )

        constraint, partner = partners[0]
        partner_position = index.get(partner)
        if partner_position is None:
            _warn_missing_partner(entry, partner)
            visited[position] = True
            continue
        if partner_position == position:
            visited[position] = True
            if isinstance(constraint, HermiticityConstraint):
                required_position = add_required(position)
                add_real_param(position, sign=1.0, required_position=required_position)
            continue

        representative = min(position, partner_position)
        paired = max(position, partner_position)
        visited[representative] = True
        visited[paired] = True
        required_position = add_required(representative)
        add_real_param(
            representative,
            sign=1.0,
            required_position=required_position,
        )
        add_imag_param(
            representative,
            sign=1.0,
            required_position=required_position,
        )
        active_real_param[paired] = active_real_param[representative]
        active_imag_param[paired] = active_imag_param[representative]
        if isinstance(constraint, HermiticityConstraint):
            active_real_sign[paired] = 1.0
            active_imag_sign[paired] = -1.0
        else:
            active_real_sign[paired] = -1.0
            active_imag_sign[paired] = -1.0

    required_coordinates = DensityCoordinates.from_entries(
        size=active_coordinates.size,
        keys=list(active_coordinates.keys),
        entries=tuple(required_entries),
    )

    required_count = len(required_entries)
    required_value_rows = np.asarray(
        [
            required_position + (required_count if imag else 0)
            for required_position, imag in required_param_rows
        ],
        dtype=int,
    )
    return _OrbitParametrization(
        required_coordinates=required_coordinates,
        required_value_rows=required_value_rows,
        active_real_param=active_real_param,
        active_real_sign=active_real_sign,
        active_imag_param=active_imag_param,
        active_imag_sign=active_imag_sign,
    )


def _constraint_partners(
    entry: DensityEntry,
    constraints: tuple[HermiticityConstraint | ParticleHoleConstraint, ...],
) -> list[tuple[HermiticityConstraint | ParticleHoleConstraint, DensityEntry]]:
    partners = []
    for constraint in constraints:
        partner = constraint.partner(entry)
        if partner is not None:
            partners.append((constraint, partner))
    return partners


def _raise_if_dense_basis_too_large(value_count: int, *, family: str) -> None:
    real_rows = 2 * int(value_count)
    estimated_bytes = 8 * real_rows * real_rows
    if estimated_bytes <= _DENSE_BASIS_LIMIT_BYTES:
        return
    estimated_gib = estimated_bytes / 1024**3
    limit_gib = _DENSE_BASIS_LIMIT_BYTES / 1024**3
    raise MemoryError(
        f"{family} SCF space with spatial symmetries would need a dense basis "
        f"with about {estimated_gib:.1f} GiB of float64 storage, above the "
        f"{limit_gib:.1f} GiB safety limit. The compact orbit parametrization "
        "currently supports only the no-spatial-symmetry case."
    )
