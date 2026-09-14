"""Public density calculations for Hamiltonian dictionaries and models."""

from dataclasses import replace

from meanfi.errors import (
    ToleranceFunction,
    default_solver_tolerances,
    resolve_error_tolerances,
)
from meanfi.density.density import evaluate_density
from meanfi.density.problem import build_density_problem
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.results import DensityResult
from meanfi.space.coordinates import DensityCoordinates
from meanfi.space.space import ActiveSCFSpace
from meanfi.tb.validate import validate_tb_dict, validate_hermiticity


def _density_problem(
    h,
    *,
    kT,
    keys,
    coordinates,
    interaction,
    spatial_symmetries,
    mean_field,
    integration,
    tolerances,
):
    electron_ndof = None
    if isinstance(h, Model):
        model = h
        if interaction is not None or spatial_symmetries:
            raise ValueError("set interaction and spatial_symmetries on the Model")
        if keys is None and coordinates is None:
            coordinates = model.scf_space.required_coordinates
        kT = model.kT if kT is None else kT
        electron_ndof = model._ndof if model.superconducting else None
        h = model.hamiltonian_from_meanfield(mean_field)
    elif mean_field is not None:
        raise ValueError("mean_field requires a Model input")
    if sum(item is not None for item in (keys, coordinates, interaction)) != 1:
        raise ValueError(
            "exactly one of keys, coordinates, or interaction must be provided"
        )
    if spatial_symmetries and interaction is None:
        raise ValueError(
            "spatial_symmetries requires interaction or a configured Model"
        )
    if interaction is not None:
        validate_tb_dict(interaction)
        validate_hermiticity(interaction)
        coordinates = ActiveSCFSpace.from_interaction(
            interaction, spatial_symmetries=spatial_symmetries
        ).required_coordinates
    validate_tb_dict(h)
    validate_hermiticity(h)
    return build_density_problem(
        h,
        kT=0.0 if kT is None else kT,
        keys=list(coordinates.keys) if coordinates is not None else keys,
        density_coordinates=coordinates,
        integration=integration,
        tolerances=tolerances,
        electron_ndof=electron_ndof,
    )


def density_matrix_at_mu(
    h,
    mu: float,
    kT: float | None = None,
    keys: list[tuple[int, ...]] | None = None,
    *,
    mean_field=None,
    coordinates: DensityCoordinates | None = None,
    interaction=None,
    spatial_symmetries=(),
    integration: IntegrationMethod | None = None,
    tol: float = 1e-3,
    tolerance_policy: ToleranceFunction = default_solver_tolerances,
) -> DensityResult:
    """Compute density at fixed chemical potential.

    A Model supplies temperature, normal/BdG structure and required entries.
    ``mean_field`` optionally adds a correction to its bare Hamiltonian.
    For a Hamiltonian dictionary, supply exactly one of ``keys`` (full blocks),
    ``coordinates`` (explicit entries) or ``interaction`` (required entries).
    Temperature defaults to zero for dictionaries.
    """
    problem = _density_problem(
        h,
        kT=kT,
        keys=keys,
        coordinates=coordinates,
        interaction=interaction,
        spatial_symmetries=spatial_symmetries,
        mean_field=mean_field,
        integration=integration,
        tolerances=resolve_error_tolerances(tol, tolerance_policy),
    )
    return evaluate_density(problem, mu=mu)


def density_matrix(
    h,
    filling: float | None = None,
    kT: float | None = None,
    keys: list[tuple[int, ...]] | None = None,
    *,
    mean_field=None,
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
    """Compute density at fixed electron filling per unit cell.

    A Model supplies filling, temperature and the normal/BdG density layout.
    Override ``keys`` to request complete blocks for analysis or ``to_tb()``.
    For a Hamiltonian dictionary, supply filling and exactly one selection mode:
    ``keys``, ``coordinates`` or ``interaction``. See ``density_matrix_at_mu``.
    """
    if filling is None:
        if not isinstance(h, Model):
            raise ValueError("filling is required for a Hamiltonian dictionary")
        filling = h.filling
    tolerances = resolve_error_tolerances(tol, tolerance_policy)
    if filling_tol is not None:
        tolerances = replace(tolerances, filling_residual=float(filling_tol))
    problem = _density_problem(
        h,
        kT=kT,
        keys=keys,
        coordinates=coordinates,
        interaction=interaction,
        spatial_symmetries=spatial_symmetries,
        mean_field=mean_field,
        integration=integration,
        tolerances=tolerances,
    )
    return evaluate_density(
        problem,
        filling=filling,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
    )
