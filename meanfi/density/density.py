"""Evaluate a normalized density problem with one of the two integration methods."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from meanfi.errors import ConvergenceError

from meanfi.density.integrate.methods import FermiSimplex
from meanfi.density.integrate.simplex import solve_simplex
from meanfi.density.integrate.periodic import solve_periodic
from meanfi.results import DensityResult
from meanfi.density.problem import DensityProblem


def evaluate_density(
    problem: DensityProblem,
    *,
    mu: float | None = None,
    filling: float | None = None,
    mu_tol: float = 1e-10,
    max_charge_evaluations: int | None = None,
    mu_guess: float = 0.0,
) -> DensityResult:
    if (mu is None) == (filling is None):
        raise ValueError("Provide exactly one of mu and filling")
    if mu is not None and not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if not math.isfinite(mu_tol) or mu_tol <= 0:
        raise ValueError("mu_tol must be positive and finite")
    if max_charge_evaluations is not None and (
        isinstance(max_charge_evaluations, bool)
        or not isinstance(max_charge_evaluations, Integral)
        or max_charge_evaluations <= 0
    ):
        raise ValueError(
            "max_charge_evaluations must be a positive integer when provided"
        )
    try:
        evaluate = (
            solve_simplex
            if isinstance(problem.integration, FermiSimplex)
            else solve_periodic
        )
        return evaluate(
            problem,
            mu=mu,
            filling=filling,
            mu_tol=mu_tol,
            max_charge_evaluations=max_charge_evaluations,
            mu_guess=mu_guess,
        )
    except (ConvergenceError, NotImplementedError):
        raise
    except (RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ConvergenceError(str(exc)) from exc
