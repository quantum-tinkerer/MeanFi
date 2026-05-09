"Mean-field tight-binding solver."

from __future__ import annotations

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .results import (
    AdaptiveQuadratureInfo,
    AdaptiveSimplexInfo,
    DensityMatrixResult,
    SCFInfo,
    SolverResult,
    UniformGridInfo,
)
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
from .observables import expectation_value
from .scf.engine import NoConvergence
from .scf.methods import AndersonMixing, LinearMixing, SCFMethod
from .scf.scf import solver
from .tb.tb import (
    add_tb,
    fermi_energy,
    generate_tb_keys,
    guess_tb,
    ifftn_to_tb,
    kgrid_to_tb,
    scale_tb,
    tb_to_kfunc,
    tb_to_kgrid,
    tb_to_tight_binding_model,
    tb_to_vertex_cache,
)


def density_matrix_at_mu(
    h,
    mu: float,
    kT: float = DEFAULT_KT,
    keys: list[tuple[int, ...]] | None = None,
    *,
    integration: IntegrationMethod | None = None,
) -> DensityMatrixResult:
    """Compute the real-space density matrix at a fixed chemical potential."""

    if keys is None:
        raise ValueError("keys must be provided")
    return _solve_density_matrix_at_mu(
        h,
        mu=mu,
        kT=kT,
        keys=keys,
        integration=integration,
    )


def density_matrix(
    h,
    filling: float,
    kT: float = DEFAULT_KT,
    keys: list[tuple[int, ...]] | None = None,
    *,
    integration: IntegrationMethod | None = None,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_charge_evaluations: int | None = None,
) -> DensityMatrixResult:
    """Compute the fixed-filling real-space density matrix."""

    if keys is None:
        raise ValueError("keys must be provided")
    return _solve_density_matrix_fixed_filling(
        h,
        filling=filling,
        kT=kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
    )


__all__ = [
    "AdaptiveQuadrature",
    "AdaptiveQuadratureInfo",
    "AdaptiveSimplex",
    "AdaptiveSimplexInfo",
    "AndersonMixing",
    "BdGMatrixFunction",
    "DensityMatrixResult",
    "DirectDiagonalization",
    "IntegrationMethod",
    "LinearMixing",
    "Model",
    "NoConvergence",
    "RationalFOE",
    "SCFInfo",
    "SCFMethod",
    "SolverResult",
    "UniformGrid",
    "UniformGridInfo",
    "__version__",
    "__version_tuple__",
    "add_tb",
    "density_matrix",
    "density_matrix_at_mu",
    "expectation_value",
    "fermi_dirac",
    "fermi_energy",
    "generate_tb_keys",
    "guess_tb",
    "ifftn_to_tb",
    "kgrid_to_tb",
    "meanfield",
    "scale_tb",
    "solver",
    "tb_to_kfunc",
    "tb_to_kgrid",
    "tb_to_tight_binding_model",
    "tb_to_vertex_cache",
]
