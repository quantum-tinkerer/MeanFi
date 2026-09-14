"""Density inputs, normalized once for direct evaluation or SCF reuse."""

from __future__ import annotations

from dataclasses import dataclass

from meanfi.errors import ErrorTolerances, resolve_integration_tolerances
from meanfi.density.integrate.common import validate_integration_method
from meanfi.density.integrate.defaults import select_default_integration
from meanfi.density.integrate.methods import IntegrationMethod, PeriodicGrid
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates
from meanfi.tb.ops import _tb_type
from meanfi.tb.validate import (
    normalize_keys,
    tb_dimension,
    tb_orbital_count,
    require_zero_dim_local_key_only,
)


@dataclass(frozen=True)
class DensityProblem:
    hamiltonian: _tb_type
    kT: float
    integration: IntegrationMethod
    tolerances: ErrorTolerances
    density_coordinates: DensityCoordinates
    include_band_energy: bool = False
    electron_ndof: int | None = None


def build_density_problem(
    hamiltonian: _tb_type,
    *,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
    density_coordinates: DensityCoordinates | None = None,
    include_band_energy: bool = False,
    electron_ndof: int | None = None,
) -> DensityProblem:
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
    integration = integration or select_default_integration(
        hamiltonian, kT=kT, superconducting=electron_ndof is not None
    )
    integration, tolerances = resolve_integration_tolerances(integration, tolerances)
    validate_integration_method(integration, kT=kT)
    if electron_ndof is not None and not isinstance(integration, PeriodicGrid):
        raise ValueError(
            "Superconducting density requires PeriodicGrid; at kT == 0 specify nk"
        )
    keys = normalize_keys(hamiltonian, keys)
    size = tb_orbital_count(hamiltonian)
    coordinates = density_coordinates or full_density_coordinates(keys, size=size)
    if coordinates.size != size:
        raise ValueError(
            "density coordinate matrix size must match the Hamiltonian shape"
        )
    return DensityProblem(
        hamiltonian,
        kT,
        integration,
        tolerances,
        coordinates,
        include_band_energy,
        electron_ndof,
    )
