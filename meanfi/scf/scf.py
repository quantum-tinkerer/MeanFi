"""Top-level self-consistent-field pipeline."""

from __future__ import annotations

from meanfi.errors import (
    ErrorTolerances,
    ToleranceFunction,
    default_solver_tolerances,
    resolve_error_tolerances,
)

from meanfi.density.problem import build_density_problem
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.results import SCFResult
from meanfi.scf.engine import run_scf_loop
from meanfi.scf.methods import AndersonMixing, EnergyDIIS, LinearMixing, SCFMethod
from meanfi.scf.problem import SCFProblem
from meanfi.tb.ops import _tb_type


def solver(
    model: Model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod | None = None,
    scf: SCFMethod | None = None,
    tol: float | ErrorTolerances = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
    max_charge_evaluations: int | None = None,
    verbose: bool = False,
    compute_free_energy: bool = False,
) -> SCFResult:
    """Run mean-field update -> density update -> SCF mixing.

    ``tol`` accepts a number or an explicit ErrorTolerances record.

    Entropy and free energy are omitted by default. ``compute_free_energy=True``
    requests one final evaluation, including valid partial results on failure.
    SCF uses internal energy.
    """

    tolerances = resolve_error_tolerances(tol, tolerance_policy)
    resolved_scf = scf if scf is not None else EnergyDIIS()
    if not isinstance(resolved_scf, (AndersonMixing, EnergyDIIS, LinearMixing)):
        raise TypeError("scf must be LinearMixing, EnergyDIIS, or AndersonMixing")

    density_problem = build_density_problem(
        model.hamiltonian_from_meanfield(),
        kT=model.kT,
        keys=model.required_coordinates.keys,
        integration=integration,
        tolerances=tolerances,
        density_coordinates=model.required_coordinates,
        electron_ndof=model._ndof if model.superconducting else None,
    )
    problem = SCFProblem(model, density_problem, max_charge_evaluations)
    return run_scf_loop(
        guess,
        scf=resolved_scf,
        problem=problem,
        verbose=verbose,
        compute_free_energy=compute_free_energy,
    )


__all__ = ["solver"]
