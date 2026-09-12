"""Public result assembly for density pipeline payloads."""

from __future__ import annotations

from meanfi.density.internal import DensityEvaluation
from meanfi.density.problem import DensityPlan, DensityProblem
from meanfi.results import DensityResult


def wrap_density_evaluation(
    problem: DensityProblem,
    plan: DensityPlan,
    evaluation: DensityEvaluation,
    *,
    preserve_layout: bool = False,
) -> DensityResult:
    """Expose evaluated values without inventing values outside their layout."""

    del plan
    density = evaluation.density
    if not preserve_layout:
        density = density.select_keys(problem.requested_keys)
    return DensityResult(
        coordinates=density.coordinates,
        values=density.values,
        mu=evaluation.mu,
        filling=evaluation.filling,
        errors=evaluation.errors,
        statistics=evaluation.statistics,
    )


__all__ = ["wrap_density_evaluation"]
