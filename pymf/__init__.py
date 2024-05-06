"Mean-field tight-binding solver"

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .mf import construct_density_matrix
from .solvers import solver
from .model import Model
from .tb.tb import add_tb, scale_tb
from .tb.utils import generate_guess


__all__ = [
    "solver",
    "Model",
    "add_tb",
    "scale_tb",
    "generate_guess",
    "construct_density_matrix",
    "__version__",
    "__version_tuple__",
]
