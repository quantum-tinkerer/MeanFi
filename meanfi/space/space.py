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
    ActiveCoordinateSupport,
    bdg_active_support,
    normal_active_support,
)
from meanfi.space.symmetry import HermiticityConstraint, ParticleHoleConstraint
from meanfi.tb.ops import _tb_type, is_sparse_like

if TYPE_CHECKING:
    from meanfi.model import Model


_DENSE_BASIS_LIMIT_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class _OrbitParametrization:
    required_coordinates: DensityCoordinates
    required_real_rows: np.ndarray
    required_value_rows: np.ndarray
    active_real_param: np.ndarray
    active_real_sign: np.ndarray
    active_imag_param: np.ndarray
    active_imag_sign: np.ndarray
    parameter_count: int


@dataclass(frozen=True)
class ActiveSCFSpace:
    """Minimal real variables used by one mean-field SCF problem."""

    active_coordinates: DensityCoordinates
    required_coordinates: DensityCoordinates
    required_real_rows: np.ndarray
    required_value_rows: np.ndarray
    interaction_keys: list[tuple[int, ...]]
    density_keys: list[tuple[int, ...]]
    onsite: tuple[int, ...]
    basis: np.ndarray | None = None
    required_to_params: np.ndarray | None = None
    active_real_param: np.ndarray | None = None
    active_real_sign: np.ndarray | None = None
    active_imag_param: np.ndarray | None = None
    active_imag_sign: np.ndarray | None = None
    parameter_count: int | None = None

    @property
    def active_entries(self) -> tuple[DensityEntry, ...]:
        return self.active_coordinates.entries

    @property
    def num_params(self) -> int:
        if self.parameter_count is not None:
            return int(self.parameter_count)
        if self.basis is None:  # pragma: no cover - constructors provide one path
            raise ValueError("ActiveSCFSpace has no parametrization")
        return int(self.basis.shape[1])

    @classmethod
    def normal(cls, model: Model) -> ActiveSCFSpace:
        support = normal_active_support(model)
        entries = support.coordinates.entries
        if not getattr(model, "spatial_symmetries", ()):
            parametrization = _orbit_parametrization(
                support.coordinates,
                constraints=(HermiticityConstraint(),),
            )
            return cls._from_orbit_support(support, parametrization)
        _raise_if_dense_basis_too_large(len(entries), family="normal")
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
        constraints = (
            HermiticityConstraint(electron_ndof=model._ndof),
            ParticleHoleConstraint(model._ndof),
        )
        if not getattr(model, "spatial_symmetries", ()):
            parametrization = _orbit_parametrization(
                support.coordinates,
                constraints=constraints,
            )
            return cls._from_orbit_support(support, parametrization)
        _raise_if_dense_basis_too_large(len(entries), family="bdg")
        basis = OrbitReducer(entries).basis(constraints)
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
        sample_basis = np.asarray(basis, dtype=float)[selected.active_real_rows, :]
        required_to_params = (
            np.linalg.inv(sample_basis)
            if sample_basis.size
            else np.zeros((0, 0), dtype=float)
        )
        return cls(
            active_coordinates=support.coordinates,
            required_coordinates=selected.coordinates,
            required_real_rows=selected.active_real_rows,
            required_value_rows=selected.value_real_rows,
            interaction_keys=support.interaction_keys,
            density_keys=support.density_keys,
            onsite=support.onsite,
            basis=np.asarray(basis, dtype=float),
            required_to_params=required_to_params,
            parameter_count=int(basis.shape[1]),
        )

    @classmethod
    def _from_orbit_support(
        cls,
        support: ActiveCoordinateSupport,
        parametrization: _OrbitParametrization,
    ) -> ActiveSCFSpace:
        return cls(
            active_coordinates=support.coordinates,
            required_coordinates=parametrization.required_coordinates,
            required_real_rows=parametrization.required_real_rows,
            required_value_rows=parametrization.required_value_rows,
            interaction_keys=support.interaction_keys,
            density_keys=support.density_keys,
            onsite=support.onsite,
            active_real_param=parametrization.active_real_param,
            active_real_sign=parametrization.active_real_sign,
            active_imag_param=parametrization.active_imag_param,
            active_imag_sign=parametrization.active_imag_sign,
            parameter_count=parametrization.parameter_count,
        )

    def required_realspace_entries(self) -> tuple[DensityEntry, ...]:
        return self.required_coordinates.entries

    def required_density_coordinates_for(
        self, tb: _tb_type
    ) -> DensityCoordinates | None:
        if self.required_coordinates.value_count == 0:
            return None
        if any(is_sparse_like(matrix) for matrix in tb.values()):
            return self.required_coordinates
        return None

    def params_from_required_entries(self, values: np.ndarray) -> np.ndarray:
        real_values = complex_to_real(values)
        if real_values.size != 2 * self.required_coordinates.value_count:
            raise ValueError("values do not match required real-space entries")
        if self.num_params == 0:
            return np.empty(0, dtype=float)
        if self.required_to_params is None:
            return np.asarray(real_values[self.required_value_rows], dtype=float)
        return np.asarray(
            self.required_to_params @ real_values[self.required_value_rows],
            dtype=float,
        )

    def params_from_meanfield_input(self, rho: _tb_type) -> np.ndarray:
        return self.params_from_required_entries(
            self.required_coordinates.values_from_tb(rho)
        )

    def meanfield_input_from_params(self, params: np.ndarray) -> _tb_type:
        params = np.asarray(params, dtype=float).reshape(-1)
        if params.size != self.num_params:
            raise ValueError("params has the wrong length for this active SCF space")
        if self.basis is None:
            return self._meanfield_input_from_orbit_params(params)
        real_values = self.basis @ params
        return self.active_coordinates.values_to_tb(real_to_complex(real_values))

    def project_meanfield_input(self, rho: _tb_type) -> _tb_type:
        return self.meanfield_input_from_params(self.params_from_meanfield_input(rho))

    def _meanfield_input_from_orbit_params(self, params: np.ndarray) -> _tb_type:
        if (
            self.active_real_param is None
            or self.active_real_sign is None
            or self.active_imag_param is None
            or self.active_imag_sign is None
        ):  # pragma: no cover - constructors provide complete orbit metadata
            raise ValueError("ActiveSCFSpace has incomplete orbit parametrization")

        values = np.zeros(self.active_coordinates.value_count, dtype=complex)
        real_mask = self.active_real_param >= 0
        values[real_mask] += (
            self.active_real_sign[real_mask]
            * params[self.active_real_param[real_mask]]
        )
        imag_mask = self.active_imag_param >= 0
        values[imag_mask] += (
            1j
            * self.active_imag_sign[imag_mask]
            * params[self.active_imag_param[imag_mask]]
        )
        return self.active_coordinates.values_to_tb(values)


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
    required_active_rows: list[int] = []
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
        required_active_rows.append(position)
        parameter_count += 1

    def add_imag_param(position: int, *, sign: float, required_position: int) -> None:
        nonlocal parameter_count
        active_imag_param[position] = parameter_count
        active_imag_sign[position] = sign
        required_param_rows.append((required_position, True))
        required_active_rows.append(value_count + position)
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
        allow_empty=True,
    )
    if required_coordinates is None:  # pragma: no cover - allow_empty guarantees this
        raise ValueError("Required density coordinates unexpectedly missing")

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
        required_real_rows=np.asarray(required_active_rows, dtype=int),
        required_value_rows=required_value_rows,
        active_real_param=active_real_param,
        active_real_sign=active_real_sign,
        active_imag_param=active_imag_param,
        active_imag_sign=active_imag_sign,
        parameter_count=parameter_count,
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
