"""Top-level self-consistent-field pipeline."""

from __future__ import annotations

from dataclasses import replace

from meanfi.errors import (
    ToleranceFunction,
    default_solver_tolerances,
    resolve_error_tolerances,
)

from meanfi.density.problem import build_density_problem
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.results import SCFResult
from meanfi.scf.engine import run_scf_loop
from meanfi.scf.methods import EnergyDIIS, SCFMethod
from meanfi.scf.problem import SCFProblem
from meanfi.tb.ops import _tb_type


def solver(
    model: Model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod | None = None,
    scf: SCFMethod | None = None,
    scf_tol: float | None = None,
    tol: float = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_charge_evaluations: int | None = None,
    verbose: bool = False,
    compute_free_energy: bool = True,
) -> SCFResult:
    """Run mean-field update -> density update -> SCF mixing.

    Entropy is computed once at termination by default, including valid partial
    results on failure. Set ``compute_free_energy=False`` to skip that final
    evaluation; entropy and free energy remain None. SCF uses internal energy.
    """

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
    resolved_scf = scf if scf is not None else EnergyDIIS()
    if not isinstance(resolved_scf, SCFMethod):
        raise TypeError("scf must be an SCFMethod instance")

    density_problem = build_density_problem(
        model.hamiltonian_from_meanfield(),
        kT=model.kT,
        keys=model.required_coordinates.keys,
        integration=integration,
        tolerances=tolerances,
        density_coordinates=model.required_coordinates,
        electron_ndof=model._ndof if model.superconducting else None,
    )
    problem = SCFProblem(model, density_problem, mu_tol, max_charge_evaluations)
    return run_scf_loop(
        guess,
        scf=resolved_scf,
        problem=problem,
        verbose=verbose,
        compute_free_energy=compute_free_energy,
    )


__all__ = ["solver"]
