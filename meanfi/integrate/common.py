from __future__ import annotations

import numpy as np

from meanfi.results import (
    AdaptiveQuadratureInfo,
    AdaptiveSimplexInfo,
    DensityIntegrationInfo,
    DensityMatrixResult,
    FixedFillingInfo,
    UniformGridInfo,
)
from meanfi.tb.validate import (
    normalize_keys,
    tb_dimension,
    tb_orbital_count,
    zero_key,
)
from meanfi.tb.ops import _tb_type

from meanfi.state.support import require_supported_workspace_precision
from .methods import AdaptiveQuadrature, AdaptiveSimplex, IntegrationMethod, UniformGrid


def validate_integration_method(integration: IntegrationMethod, *, kT: float) -> None:
    require_supported_workspace_precision(integration)
    if kT < 0:
        raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")
    if isinstance(integration, AdaptiveSimplex):
        if kT != 0:
            raise ValueError("AdaptiveSimplex requires kT == 0")
        return
    if isinstance(integration, AdaptiveQuadrature):
        if kT <= 0:
            raise ValueError("AdaptiveQuadrature requires kT > 0")
        return
    if isinstance(integration, UniformGrid):
        if kT < 0:
            raise ValueError("UniformGrid requires kT >= 0")
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


def trim_density_matrix(
    density_matrix: _tb_type,
    *,
    keys: list[tuple[int, ...]],
) -> _tb_type:
    sample = next(iter(density_matrix.values()))
    zeros = np.zeros_like(sample)
    return {key: np.array(density_matrix.get(key, zeros), copy=True) for key in keys}


def trim_density_matrix_error(
    density_matrix_error: _tb_type | None,
    *,
    keys: list[tuple[int, ...]],
) -> _tb_type | None:
    if density_matrix_error is None:
        return None
    sample = next(iter(density_matrix_error.values()))
    zeros = np.zeros_like(sample)
    return {
        key: np.array(density_matrix_error.get(key, zeros), copy=True) for key in keys
    }


def local_density_filling(
    density_matrix: _tb_type,
    *,
    local_key: tuple[int, ...],
) -> float:
    return float(np.trace(density_matrix[local_key]).real)


def effective_filling_tol(
    integration: IntegrationMethod,
    *,
    hamiltonian: _tb_type,
    filling_tol: float | None,
) -> float:
    if filling_tol is not None:
        if filling_tol <= 0:
            raise ValueError("filling_tol must be positive when provided")
        return float(filling_tol)

    if isinstance(integration, (AdaptiveSimplex, AdaptiveQuadrature, UniformGrid)):
        return float(
            0.1 * tb_orbital_count(hamiltonian) * integration.density_matrix_tol
        )

    raise ValueError("UniformGrid requires an implicit grid-resolved filling target")


def translate_adaptive_info(
    integration: AdaptiveSimplex | AdaptiveQuadrature,
    raw_info: DensityIntegrationInfo | FixedFillingInfo,
):
    info_type = (
        AdaptiveSimplexInfo
        if isinstance(integration, AdaptiveSimplex)
        else AdaptiveQuadratureInfo
    )
    return info_type(
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
    )


def uniform_grid_info(
    *,
    integration: UniformGrid,
    hamiltonian: _tb_type,
    n_kernel_evals: int | None = None,
    n_evaluator_evals: int | None = None,
    charge_evaluations: int | None = None,
    charge_integration_calls: int | None = None,
    density_integration_calls: int | None = None,
    error_estimate_available: bool = False,
) -> UniformGridInfo:
    ndim = tb_dimension(hamiltonian)
    n_kpoints = 1 if ndim == 0 else int(integration.nk**ndim)
    return UniformGridInfo(
        nk=int(integration.nk),
        n_kpoints=n_kpoints,
        unique_evals=n_kpoints,
        n_kernel_evals=n_kpoints if n_kernel_evals is None else int(n_kernel_evals),
        n_evaluator_evals=(
            n_kpoints if n_evaluator_evals is None else int(n_evaluator_evals)
        ),
        charge_evaluations=(
            None if charge_evaluations is None else int(charge_evaluations)
        ),
        charge_integration_calls=(
            None if charge_integration_calls is None else int(charge_integration_calls)
        ),
        density_integration_calls=(
            None
            if density_integration_calls is None
            else int(density_integration_calls)
        ),
        error_estimate_available=bool(error_estimate_available),
    )


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
) -> DensityMatrixResult:
    trimmed_density_matrix = trim_density_matrix(density_matrix, keys=keys)
    trimmed_density_matrix_error = trim_density_matrix_error(
        density_matrix_error,
        keys=keys,
    )
    filling_residual = (
        None if target_filling is None else abs(float(filling) - float(target_filling))
    )
    return DensityMatrixResult(
        density_matrix=trimmed_density_matrix,
        density_matrix_error=trimmed_density_matrix_error,
        mu=float(mu),
        filling=float(filling),
        target_filling=None if target_filling is None else float(target_filling),
        filling_residual=filling_residual,
        integration=integration,
        info=info,
    )


def wrap_adaptive_result(
    *,
    density_matrix: _tb_type,
    density_matrix_error: _tb_type | None,
    raw_info: DensityIntegrationInfo | FixedFillingInfo,
    mu: float,
    filling: float,
    target_filling: float | None,
    integration: AdaptiveSimplex | AdaptiveQuadrature,
    keys: list[tuple[int, ...]],
) -> DensityMatrixResult:
    public_info = translate_adaptive_info(integration, raw_info)
    error = density_matrix_error if public_info.error_estimate_available else None
    return wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=error,
        mu=mu,
        filling=filling,
        target_filling=target_filling,
        integration=integration,
        info=public_info,
        keys=keys,
    )


def retarget_result_keys(
    result: DensityMatrixResult,
    *,
    keys: list[tuple[int, ...]],
) -> DensityMatrixResult:
    if list(result.density_matrix) == list(keys):
        return result
    return DensityMatrixResult(
        density_matrix=trim_density_matrix(result.density_matrix, keys=keys),
        density_matrix_error=trim_density_matrix_error(
            result.density_matrix_error,
            keys=keys,
        ),
        mu=result.mu,
        filling=result.filling,
        target_filling=result.target_filling,
        filling_residual=result.filling_residual,
        integration=result.integration,
        info=result.info,
    )
