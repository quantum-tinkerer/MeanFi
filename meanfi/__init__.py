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
from .solvers import SolverInfo, solver
from .tb.tb import add_tb, scale_tb
from .tb.transforms import tb_to_kfunc
from .tb.utils import guess_tb


__all__ = [
    "DensityIntegrationInfo",
    "FixedFillingInfo",
    "SolverInfo",
    "Model",
    "add_tb",
    "density_matrix",
    "density_matrix_at_mu",
    "expectation_value",
    "fermi_dirac",
    "guess_tb",
    "meanfield",
    "scale_tb",
    "solver",
    "tb_to_kfunc",
    "__version__",
    "__version_tuple__",
]
