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

from .results import DensityEntries, DensityResult, SCFIteration, SCFResult
from .density.density import evaluate_density as _evaluate_density
from .density.problem import build_normal_problem as _build_normal_problem
from .density.integrate.defaults import DEFAULT_KT
from .density.integrate.methods import (
    AdaptiveSimplex,
    IntegrationMethod,
    PeriodicGrid,
)
from .density.kpoint.matrix_functions import (
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


def _density_selection(h, kT, keys, coordinates, interaction, spatial_symmetries):
    if sum(item is not None for item in (keys, coordinates, interaction)) != 1:
        raise ValueError(
            "exactly one of keys, coordinates, or interaction must be provided"
        )
    if interaction is not None:
        # Filling does not affect the interaction's required coordinate layout.
        coordinates = Model(
            h,
            interaction,
            filling=1.0,
            kT=kT,
            spatial_symmetries=spatial_symmetries,
        ).scf_space.required_coordinates
    if coordinates is not None:
        return list(coordinates.keys), coordinates
    return [tuple(key) for key in keys], None


def density_matrix_at_mu(
    h,
    mu: float,
    kT: float = DEFAULT_KT,
    keys: list[tuple[int, ...]] | None = None,
    *,
    coordinates: DensityCoordinates | None = None,
    interaction=None,
    spatial_symmetries=(),
    integration: IntegrationMethod | None = None,
    tol: float = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
) -> DensityResult:
    """Compute density at fixed mu using one explicit selection mode.

    Supply exactly one of ``keys``, ``coordinates``, or ``interaction``, as for
    :func:`density_matrix`.
    """

    keys, coordinates = _density_selection(
        h, kT, keys, coordinates, interaction, spatial_symmetries
    )
    tolerances = resolve_error_tolerances(tol, tolerance_policy)
    problem = _build_normal_problem(
        h,
        kT=kT,
        keys=keys,
        integration=integration,
        tolerances=tolerances,
        density_coordinates=coordinates,
    )
    return _evaluate_density(problem, mu=mu)


def density_matrix(
    h,
    filling: float,
    kT: float = DEFAULT_KT,
    keys: list[tuple[int, ...]] | None = None,
    *,
    coordinates: DensityCoordinates | None = None,
    interaction=None,
    spatial_symmetries=(),
    integration: IntegrationMethod | None = None,
    tol: float = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_charge_evaluations: int | None = None,
) -> DensityResult:
    """Compute density values on one explicit layout.

    Exactly one selection mode must be supplied: ``keys`` requests complete
    matrix blocks, ``coordinates`` requests an exact advanced layout, and
    ``interaction`` requests only the entries needed by that interaction's SCF
    space.  The latter is the efficient way to construct a reference density.
    """

    selected_keys, selected_coordinates = _density_selection(
        h, kT, keys, coordinates, interaction, spatial_symmetries
    )
    tolerances = resolve_error_tolerances(tol, tolerance_policy)
    if filling_tol is not None:
        tolerances = replace(
            tolerances,
            filling_residual=float(filling_tol),
        )
    problem = _build_normal_problem(
        h,
        kT=kT,
        keys=selected_keys,
        integration=integration,
        tolerances=tolerances,
        density_coordinates=selected_coordinates,
    )
    return _evaluate_density(
        problem,
        filling=filling,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
    )


__all__ = [
    "AdaptiveSimplex",
    "AndersonMixing",
    "EnergyDIIS",
    "ErrorTolerances",
    "ErrorValues",
    "DensityCoordinates",
    "DensityEntries",
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
    "PeriodicGrid",
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
