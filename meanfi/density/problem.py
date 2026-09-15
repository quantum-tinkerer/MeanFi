"""Density inputs, normalized once for direct evaluation or SCF reuse."""

from __future__ import annotations

from dataclasses import dataclass, replace

from meanfi.errors import ErrorTolerances, resolve_integration_tolerances
from meanfi.density.kpoint.matrix_functions import resolve_periodic_matrix_function
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
    electron_ndof: int | None = None


def build_density_problem(
    hamiltonian: _tb_type,
    *,
    kT: float,
    keys: list[tuple[int, ...]] | None = None,
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
    density_coordinates: DensityCoordinates | None = None,
    electron_ndof: int | None = None,
) -> DensityProblem:
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
    integration = integration or select_default_integration(
        hamiltonian, kT=kT, superconducting=electron_ndof is not None
    )
    validate_integration_method(integration, kT=kT)
    tolerances = resolve_integration_tolerances(integration, tolerances)
    if isinstance(integration, PeriodicGrid):
        integration = replace(
            integration,
            matrix_function=resolve_periodic_matrix_function(
                integration.matrix_function,
                hamiltonian,
                kT=kT,
                prescribed=integration.nk is not None,
            ),
        )
    if electron_ndof is not None and not isinstance(integration, PeriodicGrid):
        raise ValueError(
            "Superconducting density requires PeriodicGrid; at kT == 0 specify nk"
        )
    size = tb_orbital_count(hamiltonian)
    if density_coordinates is None:
        coordinates = full_density_coordinates(
            normalize_keys(hamiltonian, keys), size=size
        )
    else:
        normalize_keys(hamiltonian, list(density_coordinates.keys))
        coordinates = density_coordinates
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
        electron_ndof,
    )
