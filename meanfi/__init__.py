"Mean-field tight-binding solver."

from __future__ import annotations


from .errors import (
    ErrorTolerances,
    ErrorValues,
    default_solver_tolerances,
    ConvergenceError,
)


try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .results import DensityResult, IntegrationInfo, SCFIteration, SCFResult
from .density.api import density_matrix, density_matrix_at_mu
from .density.integrate.methods import (
    FermiSimplex,
    UniformGrid,
)
from .density.kpoint.matrix_functions import (
    DirectDiagonalization,
    RationalFOE,
)
from .density.kpoint.occupations import fermi_dirac
from .model import Model
from .observables import expectation_value, trial_free_energy, trial_internal_energy
from .scf.engine import NoConvergence, SolverError, SolverFailure
from .scf.methods import AndersonMixing, EnergyDIIS, LinearMixing
from .scf.scf import solver
from .space import DensityCoordinates, SpatialSymmetry
from .tb import (
    add_tb,
    fermi_energy,
    generate_tb_keys,
    ifftn_to_tb,
    kgrid_to_tb,
    scale_tb,
    tb_to_kfunc,
    tb_to_kgrid,
)


__all__ = [
    "FermiSimplex",
    "AndersonMixing",
    "EnergyDIIS",
    "ConvergenceError",
    "ErrorTolerances",
    "ErrorValues",
    "DensityCoordinates",
    "DensityResult",
    "IntegrationInfo",
    "DirectDiagonalization",
    "LinearMixing",
    "Model",
    "NoConvergence",
    "RationalFOE",
    "SCFIteration",
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
    "scale_tb",
    "solver",
    "tb_to_kfunc",
    "tb_to_kgrid",
    "trial_free_energy",
    "trial_internal_energy",
]
