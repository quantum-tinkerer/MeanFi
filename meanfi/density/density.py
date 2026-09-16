"""Evaluate a normalized density problem with one of the two integration methods."""

from __future__ import annotations

import math
from numbers import Integral
from dataclasses import replace

import numpy as np

from meanfi.errors import ConvergenceError, ErrorValues

from meanfi.density.integrate.methods import FermiSimplex
from meanfi.density.integrate.simplex import solve_simplex
from meanfi.density.integrate.periodic import solve_periodic
from meanfi.results import DensityResult, IntegrationInfo, _DensityEntries
from meanfi.density.problem import DensityProblem
from meanfi.tb.validate import tb_orbital_count


def evaluate_density(
    problem: DensityProblem,
    *,
    mu: float | None = None,
    filling: float | None = None,
    max_charge_evaluations: int | None = None,
    mu_guess: float = 0.0,
    compute_entropy: bool = False,
) -> DensityResult:
    if (mu is None) == (filling is None):
        raise ValueError("Provide exactly one of mu and filling")
    size = problem.electron_ndof or tb_orbital_count(problem.hamiltonian)
    if filling is not None and (not math.isfinite(filling) or not 0 <= filling <= size):
        raise ValueError(
            "filling must be finite and between zero and the orbital count"
        )
    if mu is not None and not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if max_charge_evaluations is not None and (
        isinstance(max_charge_evaluations, bool)
        or not isinstance(max_charge_evaluations, Integral)
        or max_charge_evaluations <= 0
    ):
        raise ValueError(
            "max_charge_evaluations must be a positive integer when provided"
        )
    if (
        filling is None
        and not problem.density_coordinates.value_count
        and not compute_entropy
    ):
        return DensityResult(
            entries=_DensityEntries(problem.density_coordinates, np.empty(0, complex)),
            mu=float(mu),
            filling=None,
            errors=ErrorValues(),
            statistics=IntegrationInfo(
                n_kpoints=0, n_kernel_evals=0, requested_nk=problem.integration.nk
            ),
            kT=problem.kT,
        )
    try:
        evaluate = (
            solve_simplex
            if isinstance(problem.integration, FermiSimplex)
            else solve_periodic
        )
        result = evaluate(
            problem,
            mu=mu,
            filling=filling,
            mu_tol=problem.tolerances.mu_tol,
            max_charge_evaluations=max_charge_evaluations,
            mu_guess=mu_guess,
            compute_entropy=compute_entropy,
        )
        return replace(result, kT=problem.kT)
    except (ConvergenceError, NotImplementedError):
        raise
    except (RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ConvergenceError(str(exc)) from exc
