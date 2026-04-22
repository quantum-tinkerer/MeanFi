"Mean-field tight-binding solver."

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .mf import (
    DensityIntegrationInfo,
    FixedFillingInfo,
    density_matrix,
    density_matrix_at_mu,
    fermi_dirac,
    meanfield,
)
from .model import Model
from .observables import expectation_value
from .solvers import NoConvergence, SolverInfo, solver
from .tb.tb import add_tb, scale_tb
from .tb.transforms import (
    ifftn_to_tb,
    kgrid_to_tb,
    tb_to_kfunc,
    tb_to_kgrid,
)
from .tb._native import tb_to_tight_binding_model, tb_to_vertex_cache
from .tb.utils import fermi_energy, generate_tb_keys, guess_tb


__all__ = [
    "DensityIntegrationInfo",
    "FixedFillingInfo",
    "SolverInfo",
    "Model",
    "NoConvergence",
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
