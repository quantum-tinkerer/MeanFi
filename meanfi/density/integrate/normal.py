"""Normal-state FermiSimplex dispatch and finite-system handling."""

from __future__ import annotations
from dataclasses import replace
import numpy as np

from meanfi.density.integrate.common import (
    local_density_filling,
    retarget_result_keys,
    wrap_adaptive_result,
)
from meanfi.density.integrate.methods import AdaptiveSimplex
from meanfi.density.integrate.simplex import (
    density_matrix_at_mu_zero_temp,
    density_matrix_zero_temp,
)
from meanfi.density.kpoint.zero_dim import (
    density_matrix_at_mu_zero_dim,
    density_matrix_zero_dim,
)
from meanfi.density.problem import DensityProblem
from meanfi.tb.ops import _tb_type, to_dense
from meanfi.tb.validate import require_zero_dim_local_key_only, tb_dimension, zero_key


def _wrap_adaptive_payload(
    request_context: DensityProblem,
    *,
    density_matrix: _tb_type,
    density_matrix_error: _tb_type | None,
    raw_info: object,
    mu: float,
    filling: float,
    target_filling: float | None,
):
    integration = request_context.integration
    assert isinstance(integration, AdaptiveSimplex)
    result = wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=mu,
        filling=filling,
        target_filling=target_filling,
        integration=integration,
        keys=request_context.solve_keys,
        density_coordinates=request_context.density_coordinates,
    )
    return retarget_result_keys(result, keys=request_context.requested_keys)


def _adaptive_simplex_at_mu(context: DensityProblem, mu: float):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveSimplex)

    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            to_dense(hamiltonian[tuple()]),
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
        )
    else:
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_temp(
            hamiltonian,
            mu=mu,
            keys=context.solve_keys,
            density_coordinates=context.density_coordinates,
            density_atol=context.tolerances.density_matrix_integration,
            charge_tol=context.tolerances.charge_integration,
            density_rtol=0.0,
            max_subdivisions=integration.max_refinements,
            num_threads=integration.num_threads,
            nk=integration.nk,
            max_points=integration.max_points,
        )

    if tb_dimension(hamiltonian) == 0:
        raw_info = replace(
            raw_info,
            charge_error=0.0 if integration.nk is None else None,
            error_estimate_available=integration.nk is None,
            requested_nk=integration.nk,
            n_kpoints=1,
            n_diagonalizations=1,
        )

    return _wrap_adaptive_payload(
        context,
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=mu,
        filling=(
            raw_info.charge
            if getattr(raw_info, "charge", None) is not None
            else local_density_filling(
                density_matrix, local_key=zero_key(tb_dimension(hamiltonian))
            )
        ),
        target_filling=None,
    )


def _adaptive_simplex_fixed_filling(
    context: DensityProblem,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveSimplex)

    resolved_filling_tol = (
        context.tolerances.filling_residual if filling_tol is None else filling_tol
    )
    resolved_charge_tol = context.tolerances.charge_integration
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_dim(
            matrix=to_dense(hamiltonian[tuple()]),
            filling=filling,
            kT=context.kT,
            keys=context.solve_keys,
            mu_guess=mu_guess,
            charge_tol=resolved_charge_tol,
            mu_xtol=mu_tol,
            max_charge_evaluations=max_charge_evaluations,
            density_atol=context.tolerances.density_matrix_integration,
            density_rtol=0.0,
        )
        if abs(raw_info.charge - filling) > resolved_filling_tol:
            raise RuntimeError(
                "The zero-dimensional spectrum cannot represent the requested filling"
            )
        if context.include_band_energy:
            raw_info = replace(
                raw_info,
                band_energy=float(
                    np.real(
                        np.trace(
                            to_dense(hamiltonian[tuple()]) @ density_matrix[tuple()]
                        )
                    )
                ),
            )
    else:
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_temp(
            hamiltonian,
            filling=filling,
            keys=context.solve_keys,
            density_coordinates=context.density_coordinates,
            charge_tol=resolved_charge_tol,
            filling_tol=resolved_filling_tol,
            density_atol=context.tolerances.density_matrix_integration,
            density_rtol=0.0,
            mu_guess=mu_guess,
            mu_xtol=mu_tol,
            max_charge_evaluations=max_charge_evaluations,
            max_subdivisions=integration.max_refinements,
            num_threads=integration.num_threads,
            nk=integration.nk,
            max_points=integration.max_points,
            include_band_energy=context.include_band_energy,
        )

    if tb_dimension(hamiltonian) == 0:
        raw_info = replace(
            raw_info,
            charge_error=0.0 if integration.nk is None else None,
            error_estimate_available=integration.nk is None,
            requested_nk=integration.nk,
            n_kpoints=1,
            n_diagonalizations=1,
        )

    return _wrap_adaptive_payload(
        context,
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
    )
