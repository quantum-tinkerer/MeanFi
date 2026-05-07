from __future__ import annotations

from adaptivesimplex import NATIVE_AVAILABLE as _ZERO_TEMP_EXT_AVAILABLE
from adaptivesimplex import density_matrix_at_mu_zero_temp as _density_matrix_at_mu_zero_temp
from adaptivesimplex import density_matrix_zero_temp as _density_matrix_zero_temp

from meanfi.tb.ops import _tb_type


def density_matrix_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    density_atol: float,
    density_rtol: float,
    max_subdivisions: int | None = None,
    refinement_depth: int = 0,
):
    return _density_matrix_at_mu_zero_temp(
        h,
        mu=mu,
        keys=keys,
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
