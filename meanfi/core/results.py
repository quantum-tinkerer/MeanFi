from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DensityIntegrationInfo:
    """Internal statistics for a single density integration at fixed chemical potential."""

    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int
    error_estimate_available: bool


@dataclass(frozen=True)
class FixedFillingInfo:
    """Internal statistics for a fixed-filling density calculation."""

    mu: float
    charge: float
    charge_error: float
    dcharge_dmu: float
    root_iterations: int
    charge_integration_calls: int
    density_integration_calls: int
    charge_n_kernel_evals: int
    density_n_kernel_evals: int
    n_kernel_evals: int
    unique_evals: int
    charge_n_evaluator_evals: int
    density_n_evaluator_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int
    charge_integral_atol: float
    density_atol: float
    density_rtol: float
    error_estimate_available: bool


@dataclass(frozen=True)
class AdaptiveSimplexInfo:
    """Public runtime metadata for adaptive zero-temperature simplicial integration."""

    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    refinements: int
    error_estimate_available: bool
    root_iterations: int | None = None
    charge_integration_calls: int | None = None
    density_integration_calls: int | None = None


@dataclass(frozen=True)
class AdaptiveQuadratureInfo:
    """Public runtime metadata for adaptive finite-temperature quadrature."""

    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    refinements: int
    error_estimate_available: bool
    root_iterations: int | None = None
    charge_integration_calls: int | None = None
    density_integration_calls: int | None = None


@dataclass(frozen=True)
class UniformGridInfo:
    """Public runtime metadata for uniform-grid zero-temperature integration."""

    nk: int
    n_kpoints: int
    unique_evals: int
    error_estimate_available: bool = False


@dataclass(frozen=True)
class SCFInfo:
    """Public runtime metadata for an SCF solve."""

    method: str
    iterations: int
    residual_norm: float
    total_charge_integration_calls: int
    total_density_integration_calls: int
    total_kernel_evals: int
    total_unique_evals: int
    total_evaluator_evals: int


@dataclass(frozen=True)
class DensityMatrixResult:
    """Public result for a density-matrix evaluation."""

    density_matrix: dict[tuple[int, ...], Any]
    density_matrix_error: dict[tuple[int, ...], Any] | None
    mu: float
    filling: float
    target_filling: float | None
    filling_residual: float | None
    integration: object
    info: AdaptiveSimplexInfo | AdaptiveQuadratureInfo | UniformGridInfo


@dataclass(frozen=True)
class SolverResult:
    """Public result for an SCF mean-field solve."""

    mf: dict[tuple[int, ...], Any]
    density_matrix_result: DensityMatrixResult
    integration: object
    scf: object
    info: SCFInfo
