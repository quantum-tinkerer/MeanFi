from __future__ import annotations

from dataclasses import dataclass

from meanfi.results import DensityMatrixResult, SCFInfo
from meanfi.scf.methods import AndersonMixing, LinearMixing, SCFMethod


@dataclass
class SCFRunState:
    iterations: int = 0
    mu: float = 0.0
    density_matrix_result: DensityMatrixResult | None = None
    residual_norm: float = float("inf")
    total_charge_integration_calls: int = 0
    total_density_integration_calls: int = 0
    total_kernel_evals: int = 0
    total_unique_evals: int = 0
    total_evaluator_evals: int = 0


def integration_counters(result: DensityMatrixResult) -> tuple[int, int, int, int, int]:
    info = result.info
    return (
        int(getattr(info, "charge_integration_calls", 0) or 0),
        int(getattr(info, "density_integration_calls", 0) or 0),
        int(getattr(info, "n_kernel_evals", 0) or 0),
        int(
            getattr(
                info,
                "unique_evals",
                getattr(info, "n_kernel_evals", getattr(info, "n_kpoints", 0)),
            )
            or 0
        ),
        int(getattr(info, "n_evaluator_evals", 0) or 0),
    )


def record_density_result(state: SCFRunState, result: DensityMatrixResult) -> None:
    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = (
        integration_counters(result)
    )
    state.density_matrix_result = result
    state.mu = result.mu
    state.total_charge_integration_calls += charge_calls
    state.total_density_integration_calls += density_calls
    state.total_kernel_evals += kernel_evals
    state.total_unique_evals += unique_evals
    state.total_evaluator_evals += evaluator_evals


def scf_method_name(scf: SCFMethod) -> str:
    if isinstance(scf, AndersonMixing):
        return "anderson_mixing"
    if isinstance(scf, LinearMixing):
        return "linear_mixing"
    return scf.__class__.__name__


def build_scf_info(
    state: SCFRunState,
    *,
    final_result: DensityMatrixResult,
    scf: SCFMethod,
    residual_norm: float,
) -> SCFInfo:
    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = (
        integration_counters(final_result)
    )
    return SCFInfo(
        method=scf_method_name(scf),
        iterations=max(1, state.iterations),
        residual_norm=residual_norm,
        total_charge_integration_calls=state.total_charge_integration_calls
        + charge_calls,
        total_density_integration_calls=state.total_density_integration_calls
        + density_calls,
        total_kernel_evals=state.total_kernel_evals + kernel_evals,
        total_unique_evals=state.total_unique_evals + unique_evals,
        total_evaluator_evals=state.total_evaluator_evals + evaluator_evals,
    )
