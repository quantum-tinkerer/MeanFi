from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from meanfi.errors import ErrorValues


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
    num_threads: int | None = None


@dataclass(frozen=True)
class FixedFillingInfo:
    """Internal statistics for a fixed-filling density calculation."""

    mu: float
    charge: float
    charge_error: float
    dcharge_dmu: float
    charge_evaluations: int
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
    num_threads: int | None = None
    band_energy: float | None = None
    band_energy_integration_calls: int = 0
    band_energy_n_kernel_evals: int = 0


@dataclass(frozen=True)
class AdaptiveSimplexInfo:
    """Internal statistics for adaptive zero-temperature simplicial integration."""

    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    refinements: int
    error_estimate_available: bool
    charge_evaluations: int | None = None
    charge_integration_calls: int | None = None
    density_integration_calls: int | None = None
    charge_error: float | None = None
    num_threads: int | None = None
    band_energy_integration_calls: int = 0
    band_energy_n_kernel_evals: int = 0


@dataclass(frozen=True)
class AdaptiveQuadratureInfo:
    """Internal statistics for adaptive finite-temperature quadrature."""

    n_kernel_evals: int
    unique_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    refinements: int
    error_estimate_available: bool
    charge_evaluations: int | None = None
    charge_integration_calls: int | None = None
    density_integration_calls: int | None = None
    charge_error: float | None = None


@dataclass(frozen=True)
class UniformGridInfo:
    """Internal statistics for uniform-grid integration."""

    nk: int
    n_kpoints: int
    unique_evals: int
    n_kernel_evals: int | None = None
    n_evaluator_evals: int | None = None
    charge_evaluations: int | None = None
    charge_integration_calls: int | None = None
    density_integration_calls: int | None = None
    charge_error: float | None = None
    error_estimate_available: bool = False


@dataclass(frozen=True)
class DensityResult:
    """A complete density matrix on the real-space keys requested by the user."""

    density_matrix: dict[tuple[int, ...], Any]
    mu: float
    filling: float
    errors: ErrorValues


@dataclass(frozen=True)
class SCFIteration:
    """Physical and numerical values from one successful SCF evaluation."""

    step: int
    mu: float
    filling: float
    total_energy: float | None
    errors: ErrorValues


@dataclass(frozen=True)
class SCFResult:
    """A self-consistent mean-field state, or the last valid partial state."""

    mean_field: dict[tuple[int, ...], Any]
    mu: float
    filling: float
    total_energy: float | None
    errors: ErrorValues
    history: tuple[SCFIteration, ...]
    converged: bool
