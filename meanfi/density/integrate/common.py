from __future__ import annotations

from dataclasses import replace

import numpy as np

from meanfi.errors import ErrorValues

from meanfi.results import (
    AdaptiveSimplexInfo,
    DensityIntegrationInfo,
    FixedFillingInfo,
)
from meanfi.density.internal import DensityEvaluation, DensitySlice
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates
from meanfi.tb.validate import (
    normalize_keys,
    tb_dimension,
    zero_key,
)
from meanfi.tb.ops import _tb_type

from meanfi.density.integrate.workspace import require_supported_workspace_precision
from .methods import (
    AdaptiveSimplex,
    IntegrationMethod,
    PeriodicGrid,
)


def validate_integration_method(integration: IntegrationMethod, *, kT: float) -> None:
    require_supported_workspace_precision(integration)
    if not np.isfinite(kT) or kT < 0:
        raise ValueError(
            "meanfi supports only finite non-negative temperatures (kT >= 0)"
        )
    if isinstance(integration, AdaptiveSimplex):
        if kT != 0:
            raise ValueError("AdaptiveSimplex requires kT == 0")
        return
    if isinstance(integration, PeriodicGrid):
        if kT == 0 and integration.nk is None:
            raise ValueError("Zero-temperature PeriodicGrid requires explicit nk")
        return
    raise TypeError("integration must be an IntegrationMethod instance")


def prepare_keys(
    hamiltonian: _tb_type,
    keys: list[tuple[int, ...]],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], tuple[int, ...]]:
    requested_keys = normalize_keys(hamiltonian, keys)
    ndim = tb_dimension(hamiltonian)
    local_key = zero_key(ndim)
    working_keys = list(requested_keys)
    if local_key not in working_keys:
        working_keys.append(local_key)
    return requested_keys, working_keys, local_key


def local_density_filling(
    density_matrix: _tb_type,
    *,
    local_key: tuple[int, ...],
) -> float:
    return float(np.trace(density_matrix[local_key]).real)


def translate_adaptive_info(integration: AdaptiveSimplex, raw_info):
    return AdaptiveSimplexInfo(
        n_kernel_evals=int(raw_info.n_kernel_evals),
        unique_evals=int(getattr(raw_info, "unique_evals", raw_info.n_kernel_evals)),
        n_evaluator_evals=int(raw_info.n_evaluator_evals),
        n_cached_nodes=int(raw_info.n_cached_nodes),
        n_leaves=int(raw_info.n_leaves),
        n_leaf_nodes=int(raw_info.n_leaf_nodes),
        refinements=int(raw_info.subdivisions),
        error_estimate_available=bool(raw_info.error_estimate_available),
        charge_evaluations=getattr(raw_info, "charge_evaluations", None),
        charge_integration_calls=getattr(raw_info, "charge_integration_calls", None),
        density_integration_calls=getattr(raw_info, "density_integration_calls", None),
        charge_error=getattr(raw_info, "charge_error", None),
        num_threads=getattr(raw_info, "num_threads", None),
        band_energy_integration_calls=getattr(
            raw_info, "band_energy_integration_calls", 0
        ),
        band_energy_n_kernel_evals=getattr(raw_info, "band_energy_n_kernel_evals", 0),
        requested_nk=integration.nk,
        n_kpoints=getattr(raw_info, "n_kpoints", None) or int(raw_info.n_leaf_nodes),
        n_diagonalizations=getattr(raw_info, "n_diagonalizations", None),
    )


def _density_error_value(density: DensitySlice) -> float | None:
    if density.errors is None:
        return None
    if density.errors.size == 0:
        return 0.0
    return float(np.max(density.errors))


def wrap_density_result(
    *,
    density_matrix: _tb_type,
    density_matrix_error: _tb_type | None,
    mu: float,
    filling: float,
    target_filling: float | None,
    integration: IntegrationMethod,
    info,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
    band_energy: float | None = None,
) -> DensityEvaluation:
    if density_coordinates is None:
        sample = next(iter(density_matrix.values()))
        density_coordinates = full_density_coordinates(
            keys,
            size=int(sample.shape[0]),
        )
    error_estimate_available = bool(getattr(info, "error_estimate_available", False))
    density = DensitySlice.from_tb(
        density_coordinates,
        density_matrix,
        density_matrix_error if error_estimate_available else None,
    )
    filling_residual = (
        None if target_filling is None else abs(float(filling) - float(target_filling))
    )
    errors = ErrorValues(
        density_matrix_integration=_density_error_value(density),
        filling_residual=filling_residual,
        charge_integration=(
            getattr(info, "charge_error", None) if error_estimate_available else None
        ),
    )
    return DensityEvaluation(
        density=density,
        mu=float(mu),
        filling=float(filling),
        integration=integration,
        statistics=info,
        errors=errors,
        band_energy=None if band_energy is None else float(band_energy),
    )


def wrap_adaptive_result(
    *,
    density_matrix: _tb_type,
    density_matrix_error: _tb_type | None,
    raw_info: DensityIntegrationInfo | FixedFillingInfo,
    mu: float,
    filling: float,
    target_filling: float | None,
    integration: AdaptiveSimplex,
    keys: list[tuple[int, ...]],
    density_coordinates: DensityCoordinates | None = None,
) -> DensityEvaluation:
    statistics = translate_adaptive_info(integration, raw_info)
    return wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        mu=mu,
        filling=filling,
        target_filling=target_filling,
        integration=integration,
        info=statistics,
        keys=keys,
        density_coordinates=density_coordinates,
        band_energy=getattr(raw_info, "band_energy", None),
    )


def retarget_result_keys(
    result: DensityEvaluation,
    *,
    keys: list[tuple[int, ...]],
) -> DensityEvaluation:
    if list(result.density.coordinates.keys) == list(keys):
        return result
    density = result.density.select_keys(keys)
    return replace(
        result,
        density=density,
        errors=replace(
            result.errors,
            density_matrix_integration=_density_error_value(density),
        ),
    )
