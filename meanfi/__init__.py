"Mean-field tight-binding solver."

from __future__ import annotations

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .core.results import (
    AdaptiveQuadratureInfo,
    AdaptiveSimplexInfo,
    DensityMatrixResult,
    SCFInfo,
    SolverResult,
    UniformGridInfo,
)
from .integrate.dispatch import (
    solve_density_matrix_at_mu as _solve_density_matrix_at_mu,
    solve_density_matrix_fixed_filling as _solve_density_matrix_fixed_filling,
)
from .integrate.methods import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    IntegrationMethod,
    UniformGrid,
)
from .integrate.matrix_functions import (
    BdGMatrixFunction,
    DirectDiagonalization,
    RationalFOE,
)
from .integrate.occupations import fermi_dirac
from .model import Model
from .normal.meanfield import meanfield
from .observables import expectation_value
from .scf.engine import NoConvergence
from .scf.methods import AndersonMixing, LinearMixing, SCFMethod
from .solvers import solver
from .tb._backend import tb_to_tight_binding_model, tb_to_vertex_cache
from .tb.tb import add_tb, scale_tb
from .tb.transforms import ifftn_to_tb, kgrid_to_tb, tb_to_kfunc, tb_to_kgrid
from .tb.utils import fermi_energy, generate_tb_keys, guess_tb


def density_matrix_at_mu(
    h,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    *,
    integration: IntegrationMethod,
) -> DensityMatrixResult:
    """Compute the real-space density matrix at a fixed chemical potential."""

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
    kT: float,
    keys: list[tuple[int, ...]],
    *,
    integration: IntegrationMethod,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_mu_iterations: int | None = None,
) -> DensityMatrixResult:
    """Compute the fixed-filling real-space density matrix."""

    return _solve_density_matrix_fixed_filling(
        h,
        filling=filling,
        kT=kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
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
