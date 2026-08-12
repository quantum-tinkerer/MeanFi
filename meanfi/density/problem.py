"""Physical density problems entering the solve pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from meanfi.errors import (
    ErrorTolerances,
    resolve_integration_tolerances,
)
from meanfi.density.integrate.common import (
    prepare_keys,
    validate_integration_method,
)
from meanfi.density.integrate.defaults import select_default_integration
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.density.internal import DensityEvaluation
from meanfi.space.coordinates import DensityCoordinates
from meanfi.space.coordinates import full_density_coordinates
from meanfi.tb.ops import _tb_type
from meanfi.tb.validate import tb_orbital_count


@dataclass(frozen=True)
class DensityProblem:
    """Physical density problem after public inputs have been normalized."""

    family: str
    hamiltonian: _tb_type
    kT: float
    integration: IntegrationMethod
    requested_keys: list[tuple[int, ...]]
    solve_keys: list[tuple[int, ...]]
    tolerances: ErrorTolerances
    density_coordinates: DensityCoordinates
    include_band_energy: bool = False


@dataclass(frozen=True)
class DensityPlan:
    """Numerical plan for evaluating a density problem."""

    integration: IntegrationMethod
    evaluate_mu: Callable[[float], DensityEvaluation]
    solve_filling: Callable[
        [float, float | None, float, int | None, float], DensityEvaluation
    ]


def build_normal_problem(
    hamiltonian: _tb_type,
    *,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
    density_coordinates: DensityCoordinates | None = None,
    include_band_energy: bool = False,
) -> DensityProblem:
    """Normalize public normal-density inputs into one pipeline problem."""

    selected_integration = (
        integration
        if integration is not None
        else select_default_integration(hamiltonian, kT=kT)
    )
    selected_integration, resolved_tolerances = resolve_integration_tolerances(
        selected_integration, tolerances
    )
    validate_integration_method(selected_integration, kT=kT)
    requested_keys, working_keys, _local_key = prepare_keys(hamiltonian, keys)
    orbital_count = tb_orbital_count(hamiltonian)
    resolved_coordinates = (
        density_coordinates
        if density_coordinates is not None
        else full_density_coordinates(
            working_keys,
            size=orbital_count,
        )
    )
    if resolved_coordinates.size != orbital_count:
        raise ValueError(
            "density coordinate matrix size must match the Hamiltonian shape"
        )
    return DensityProblem(
        family="normal",
        hamiltonian=hamiltonian,
        kT=kT,
        integration=selected_integration,
        requested_keys=requested_keys,
        solve_keys=working_keys,
        tolerances=resolved_tolerances,
        density_coordinates=resolved_coordinates,
        include_band_energy=include_band_energy,
    )


__all__ = [
    "DensityPlan",
    "DensityProblem",
    "build_normal_problem",
]
