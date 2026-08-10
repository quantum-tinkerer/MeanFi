"Mean-field tight-binding solver."

from __future__ import annotations

from dataclasses import replace

from .errors import (
    ErrorTolerances,
    ErrorValues,
    ToleranceFunction,
    default_solver_tolerances,
    resolve_error_tolerances,
)


try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .results import DensityResult, SCFIteration, SCFResult
from .density.density import (
    solve_density_matrix_at_mu as _solve_density_matrix_at_mu,
    solve_density_matrix_fixed_filling as _solve_density_matrix_fixed_filling,
)
from .density.integrate.defaults import DEFAULT_KT
from .density.integrate.methods import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    IntegrationMethod,
    UniformGrid,
)
from .density.kpoint.matrix_functions import (
    BdGMatrixFunction,
    DirectDiagonalization,
    RationalFOE,
)
from .density.kpoint.occupations import fermi_dirac
from .model import Model
from .meanfield import meanfield
from .observables import expectation_value, total_energy
from .scf.engine import NoConvergence, SolverError, SolverFailure
from .scf.methods import AndersonMixing, EnergyDIIS, LinearMixing, SCFMethod
from .scf.scf import solver
from .space import SpatialSymmetry
from .tb.tb import (
    add_tb,
    fermi_energy,
    generate_tb_keys,
    ifftn_to_tb,
    kgrid_to_tb,
    scale_tb,
    tb_to_kfunc,
    tb_to_kgrid,
)


def density_matrix_at_mu(
    h,
    mu: float,
    kT: float = DEFAULT_KT,
    keys: list[tuple[int, ...]] | None = None,
    *,
    integration: IntegrationMethod | None = None,
    tol: float = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
) -> DensityResult:
    """Compute the real-space density matrix at a fixed chemical potential."""

    if keys is None:
        raise ValueError("keys must be provided")
    tolerances = resolve_error_tolerances(tol, tolerance_policy)
    return _solve_density_matrix_at_mu(
        h,
        mu=mu,
        kT=kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
    )


def density_matrix(
    h,
    filling: float,
    kT: float = DEFAULT_KT,
    keys: list[tuple[int, ...]] | None = None,
    *,
    integration: IntegrationMethod | None = None,
    tol: float = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_charge_evaluations: int | None = None,
) -> DensityResult:
    """Compute the fixed-filling real-space density matrix."""

    if keys is None:
        raise ValueError("keys must be provided")
    tolerances = resolve_error_tolerances(tol, tolerance_policy)
    if filling_tol is not None:
        tolerances = replace(
            tolerances,
            filling_residual=float(filling_tol),
        )
    return _solve_density_matrix_fixed_filling(
        h,
        filling=filling,
        kT=kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
    )


__all__ = [
    "AdaptiveQuadrature",
    "AdaptiveSimplex",
    "AndersonMixing",
    "BdGMatrixFunction",
    "EnergyDIIS",
    "ErrorTolerances",
    "ErrorValues",
    "DensityResult",
    "DirectDiagonalization",
    "IntegrationMethod",
    "LinearMixing",
    "Model",
    "NoConvergence",
    "RationalFOE",
    "SCFIteration",
    "SCFMethod",
    "SCFResult",
    "SolverError",
    "SolverFailure",
    "SpatialSymmetry",
    "UniformGrid",
    "__version__",
    "__version_tuple__",
    "add_tb",
    "default_solver_tolerances",
    "density_matrix",
    "density_matrix_at_mu",
    "expectation_value",
    "fermi_dirac",
    "fermi_energy",
    "generate_tb_keys",
    "ifftn_to_tb",
    "kgrid_to_tb",
    "meanfield",
    "scale_tb",
    "solver",
    "tb_to_kfunc",
    "tb_to_kgrid",
    "total_energy",
]
