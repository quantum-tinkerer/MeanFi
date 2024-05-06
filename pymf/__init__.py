"Mean-field tight-binding solver"

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .mf import (
    construct_density_matrix,
    meanfield,
)
from .solvers import solver
from .model import Model
from .observables import expectation_value
from .tb.tb import add_tb, scale_tb
from .tb.transforms import kham_to_tb, tb_to_khamvector
from .tb.utils import generate_guess, calculate_fermi_energy


__all__ = [
    "solver",
    "Model",
    "expectation_value",
    "add_tb",
    "scale_tb",
    "generate_guess",
    "calculate_fermi_energy",
    "construct_density_matrix",
    "meanfield",
    "kham_to_tb",
    "tb_to_khamvector",
    "__version__",
    "__version_tuple__",
]
