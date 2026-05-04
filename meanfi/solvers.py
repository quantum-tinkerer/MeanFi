from __future__ import annotations

from meanfi.integrate.dispatch import solve_density_matrix_fixed_filling
from meanfi.integrate.defaults import select_default_integration
from meanfi.results import SolverResult
from meanfi.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.physics.meanfield import meanfield
from meanfi.scf.bdg import BdGFamilyAdapter
from meanfi.scf.engine import NoConvergence, SolverRuntime, solve_with_family_adapter
from meanfi.scf.normal import NormalFamilyAdapter
from meanfi.scf.methods import LinearMixing, SCFMethod
from meanfi.state.normal import rparams_to_tb, tb_to_rparams
from meanfi.tb.ops import _tb_type


def _density_for_hamiltonian(
    model: Model,
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
    density_entry_support=None,
):
    return solve_density_matrix_fixed_filling(
        hamiltonian,
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=mu_guess,
        density_entry_support=density_entry_support,
    )


_evaluate_density_for_hamiltonian = _density_for_hamiltonian


def solver(
    model: Model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod | None = None,
    scf: SCFMethod = LinearMixing(),
    scf_tol: float = 1e-5,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_mu_iterations: int | None = None,
) -> SolverResult:
    """Solve for the self-consistent mean-field correction."""

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
        max_mu_iterations=max_mu_iterations,
    )
    if getattr(model, "superconducting", False):
        adapter = BdGFamilyAdapter(model, runtime)
    else:
        adapter = NormalFamilyAdapter(
            model,
            runtime,
            tb_to_rparams_fn=tb_to_rparams,
            rparams_to_tb_fn=rparams_to_tb,
            meanfield_fn=meanfield,
            density_for_hamiltonian_fn=_density_for_hamiltonian,
        )
    return solve_with_family_adapter(
        model,
        guess,
        scf=scf,
        scf_tol=scf_tol,
        runtime=runtime,
        adapter=adapter,
    )


__all__ = ["NoConvergence", "solver"]
