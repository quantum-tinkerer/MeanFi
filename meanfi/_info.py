from dataclasses import dataclass


@dataclass(frozen=True)
class DensityIntegrationInfo:
    """Statistics for a single density integration at fixed chemical potential."""

    n_kernel_evals: int
    n_evaluator_evals: int
    n_cached_nodes: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int
    error_estimate_available: bool


@dataclass(frozen=True)
class FixedFillingInfo:
    """Statistics for a fixed-filling density calculation."""

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
