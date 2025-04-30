"Mean-field tight-binding solver"

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .mf import (
    density_matrix,
    meanfield,
    fermi_level,
)
from .solvers import solver
from .model import Model
from .observables import expectation_value
from .tb.tb import add_tb, scale_tb
from .tb.transforms import tb_to_kgrid, kgrid_to_tb, tb_to_kfunc
from .tb.utils import generate_tb_vals


__all__ = [
    "solver",
    "Model",
    "expectation_value",
    "add_tb",
    "scale_tb",
    "generate_tb_vals",
    "fermi_level",
    "density_matrix",
    "meanfield",
    "tb_to_kgrid",
    "tb_to_kfunc",
    "kgrid_to_tb",
    "__version__",
    "__version_tuple__",
]
