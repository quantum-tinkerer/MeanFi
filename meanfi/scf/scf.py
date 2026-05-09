"""Top-level self-consistent-field pipeline."""

from __future__ import annotations

from meanfi.density.integrate.defaults import select_default_integration
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.results import SolverResult
from meanfi.scf.bdg import build_bdg_scf_problem
from meanfi.scf.engine import SolverRuntime, run_scf_loop
from meanfi.scf.methods import AndersonMixing, SCFMethod
from meanfi.scf.normal import build_normal_scf_problem
from meanfi.tb.ops import _tb_type


def solver(
    model: Model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod | None = None,
    scf: SCFMethod = AndersonMixing(),
    scf_tol: float = 1e-3,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_charge_evaluations: int | None = None,
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
    runtime = SolverRuntime(
        integration=resolved_integration,
        filling_tol=filling_tol,
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
        scf=scf,
        scf_tol=scf_tol,
        problem=problem,
    )


__all__ = ["solver"]
