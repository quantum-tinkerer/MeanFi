"Mean-field tight-binding solver."

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .mf import (
    DensityMatrixResult,
    density_matrix,
    density_matrix_at_mu,
    fermi_dirac,
    meanfield,
)
from ._info import (
    AdaptiveQuadratureInfo,
    AdaptiveSimplexInfo,
    SCFInfo,
    SimplexGridInfo,
    SolverResult,
    UniformGridInfo,
)
from .integration import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    IntegrationMethod,
    SimplexGrid,
    UniformGrid,
)
from .model import Model
from .observables import expectation_value
from .scf import AndersonMixing, LinearMixing, SCFMethod
from .solvers import NoConvergence, solver
from .tb.tb import add_tb, scale_tb
from .tb.transforms import (
    ifftn_to_tb,
    kgrid_to_tb,
    tb_to_kfunc,
    tb_to_kgrid,
)
from .tb._backend import tb_to_tight_binding_model, tb_to_vertex_cache
from .tb.utils import fermi_energy, generate_tb_keys, guess_tb


__all__ = [
    "AdaptiveQuadrature",
    "AdaptiveQuadratureInfo",
    "AdaptiveSimplex",
    "AdaptiveSimplexInfo",
    "AndersonMixing",
    "DensityMatrixResult",
    "IntegrationMethod",
    "LinearMixing",
    "Model",
    "NoConvergence",
    "SCFInfo",
    "SCFMethod",
    "SimplexGrid",
    "SimplexGridInfo",
    "SolverResult",
    "UniformGrid",
    "UniformGridInfo",
    "add_tb",
    "density_matrix",
    "density_matrix_at_mu",
    "expectation_value",
    "fermi_energy",
    "fermi_dirac",
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
    "__version__",
    "__version_tuple__",
]
