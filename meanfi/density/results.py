"""Public result assembly for density pipeline payloads."""

from __future__ import annotations

from meanfi.density.internal import DensityEvaluation
from meanfi.density.problem import DensityPlan, DensityProblem
from meanfi.results import DensityResult


def wrap_density_evaluation(
    problem: DensityProblem,
    plan: DensityPlan,
    evaluation: DensityEvaluation,
) -> DensityResult:
    """Expose only a complete density on the keys explicitly requested."""

    del plan
    density = evaluation.density.select_keys(problem.requested_keys)
    return DensityResult(
        density_matrix=density.to_full_tb(),
        mu=evaluation.mu,
        filling=evaluation.filling,
        errors=evaluation.errors,
    )


__all__ = ["wrap_density_evaluation"]
