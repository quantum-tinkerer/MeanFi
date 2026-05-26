from __future__ import annotations

from adaptivesimplex import NATIVE_AVAILABLE as _ZERO_TEMP_EXT_AVAILABLE
from adaptivesimplex import (
    density_matrix_at_mu_zero_temp as _density_matrix_at_mu_zero_temp,
)
from adaptivesimplex import density_matrix_zero_temp as _density_matrix_zero_temp
from adaptivesimplex import full_density_components

from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import _tb_type


def _components_from_density_coordinates(
    density_coordinates: DensityCoordinates,
) -> list[tuple[int, int, tuple[int, ...]]]:
    return [
        (int(row), int(col), tuple(int(part) for part in key))
        for key, rows, cols, _value_slice in density_coordinates.iter_key_coordinates()
        for row, col in zip(rows, cols, strict=True)
    ]


def _resolve_density_components(
    h: _tb_type,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None,
) -> list[tuple[int, int, tuple[int, ...]]]:
    if density_coordinates is not None:
        return _components_from_density_coordinates(density_coordinates)
    size = int(next(iter(h.values())).shape[0])
    return full_density_components(keys, size=size)


def density_matrix_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
    density_atol: float,
    density_rtol: float,
    max_subdivisions: int | None = None,
    refinement_depth: int = 0,
):
    return _density_matrix_at_mu_zero_temp(
        h,
        mu=mu,
        keys=keys,
        density_components=_resolve_density_components(h, keys, density_coordinates),
        density_atol=density_atol,
        density_rtol=density_rtol,
        max_subdivisions=max_subdivisions,
        refinement_depth=refinement_depth,
    )


def density_matrix_zero_temp(
    h: _tb_type,
    *,
    filling: float,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
    mu_guess: float,
    mu_xtol: float,
    max_charge_evaluations: int | None,
    max_subdivisions: int | None = None,
    refinement_depth: int = 0,
):
    return _density_matrix_zero_temp(
        h,
        filling=filling,
        keys=keys,
        density_components=_resolve_density_components(h, keys, density_coordinates),
        charge_tol=charge_tol,
        density_atol=density_atol,
        density_rtol=density_rtol,
        mu_guess=mu_guess,
        mu_xtol=mu_xtol,
        max_mu_iterations=max_charge_evaluations,
        max_subdivisions=max_subdivisions,
        refinement_depth=refinement_depth,
    )


__all__ = [
    "_ZERO_TEMP_EXT_AVAILABLE",
    "density_matrix_at_mu_zero_temp",
    "density_matrix_zero_temp",
]
