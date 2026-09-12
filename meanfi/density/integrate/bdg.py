"""BdG density uses the same periodic integration operation as normal density."""

from __future__ import annotations

import numpy as np

from meanfi.density.filling import charge_diagonal
from meanfi.density.integrate.common import validate_integration_method
from meanfi.density.integrate.methods import PeriodicGrid
from meanfi.density.integrate.periodic import solve_periodic
from meanfi.errors import default_solver_tolerances


def solve_bdg_density_fixed_filling(
    model,
    meanfield,
    *,
    keys,
    integration,
    filling_tol,
    mu_tol,
    max_charge_evaluations,
    mu_guess,
    density_coordinates=None,
    tolerances=None,
):
    if not isinstance(integration, PeriodicGrid):
        raise NotImplementedError(
            "Superconducting density requires PeriodicGrid; at kT == 0 specify nk"
        )
    validate_integration_method(integration, kT=model.kT)
    return solve_periodic(
        model.bdg_hamiltonian_from_meanfield(meanfield),
        kT=model.kT,
        keys=keys,
        integration=integration,
        filling=model.filling,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_guess=mu_guess,
        density_coordinates=density_coordinates,
        q_diag=charge_diagonal(model._ndof),
        trace_weights_diag=np.concatenate(
            [np.ones(model._ndof), np.zeros(model._ndof)]
        ),
        tolerances=default_solver_tolerances(1e-3)
        if tolerances is None
        else tolerances,
    )
