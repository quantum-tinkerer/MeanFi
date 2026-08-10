"""Top-level self-consistent-field pipeline."""

from __future__ import annotations

from dataclasses import replace

from meanfi.errors import (
    ToleranceFunction,
    default_solver_tolerances,
    resolve_error_tolerances,
    resolve_integration_tolerances,
)

from meanfi.density.integrate.defaults import select_default_integration
from meanfi.density.integrate.methods import AdaptiveSimplex, IntegrationMethod
from meanfi.model import Model
from meanfi.results import SolverResult
from meanfi.scf.bdg import build_bdg_scf_problem
from meanfi.scf.engine import SolverRuntime, run_scf_loop
from meanfi.scf.methods import AndersonMixing, EnergyDIIS, SCFMethod
from meanfi.scf.normal import build_normal_scf_problem
from meanfi.tb.ops import _tb_type


def solver(
    model: Model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod | None = None,
    scf: SCFMethod = AndersonMixing(),
    scf_tol: float | None = None,
    tol: float = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_charge_evaluations: int | None = None,
    verbose: bool = False,
) -> SolverResult:
    """Run mean-field update -> density update -> SCF mixing."""

    resolved_integration = (
        integration
        if integration is not None
        else select_default_integration(
            model.h_0,
            kT=model.kT,
            superconducting=bool(getattr(model, "superconducting", False)),
        )
    )
    tolerances = resolve_error_tolerances(tol, tolerance_policy)
    if scf_tol is not None:
        tolerances = replace(
            tolerances,
            scf_residual=float(scf_tol),
        )
    if filling_tol is not None:
        tolerances = replace(
            tolerances,
            filling_residual=float(filling_tol),
        )
    resolved_integration, tolerances = resolve_integration_tolerances(
        resolved_integration,
        tolerances,
    )
    supports_energy_diis = (
        not bool(getattr(model, "superconducting", False))
        and float(model.kT) == 0.0
        and isinstance(resolved_integration, AdaptiveSimplex)
    )
    resolved_scf = scf
    if not isinstance(resolved_scf, SCFMethod):
        raise TypeError("scf must be an SCFMethod instance")
    if isinstance(resolved_scf, EnergyDIIS) and not supports_energy_diis:
        raise ValueError(
            "EnergyDIIS currently supports normal-state zero-temperature "
            "AdaptiveSimplex calculations only"
        )

    runtime = SolverRuntime(
        integration=resolved_integration,
        tolerances=tolerances,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
    )
    problem = (
        build_bdg_scf_problem(model, runtime)
        if getattr(model, "superconducting", False)
        else build_normal_scf_problem(model, runtime)
    )
    return run_scf_loop(
        guess,
        scf=resolved_scf,
        problem=problem,
        verbose=verbose,
    )


__all__ = ["solver"]
