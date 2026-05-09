"""Public result assembly for density pipeline payloads."""

from __future__ import annotations

from meanfi.density.problem import DensityEvaluation, DensityPlan, DensityProblem
from meanfi.results import DensityMatrixResult


def wrap_density_evaluation(
    problem: DensityProblem,
    plan: DensityPlan,
    evaluation: DensityEvaluation,
    target_filling: float | None = None,
) -> DensityMatrixResult:
    """Return the public result after lower layers finish their work."""

    del problem, plan, target_filling
    return evaluation.result


__all__ = ["wrap_density_evaluation"]
