"""Density inputs, normalized once for direct evaluation or SCF reuse."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import warnings

from meanfi.errors import ErrorTolerances, resolve_integration_tolerances
from meanfi.density.kpoint.matrix_functions import DirectDiagonalization, RationalFOE
from meanfi.density.kpoint.matrix_functions.rational.common import SparseRationalLayout
from meanfi.tb.storage import prefers_sparse_storage
import numpy as np
from meanfi.density.integrate.methods import (
    FermiSimplex,
    IntegrationMethod,
    UniformGrid,
)
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates
from meanfi.tb.ops import _tb_type
from meanfi.tb.validate import (
    normalize_keys,
    tb_dimension,
    tb_orbital_count,
    require_zero_dim_local_key_only,
)


@dataclass(frozen=True)
class DensityProblem:
    hamiltonian: _tb_type
    kT: float
    integration: IntegrationMethod
    tolerances: ErrorTolerances
    density_coordinates: DensityCoordinates
    electron_ndof: int | None = None
    sparse_layout: SparseRationalLayout | None = None


def build_density_problem(
    hamiltonian: _tb_type,
    *,
    kT: float,
    keys: list[tuple[int, ...]] | None = None,
    integration: IntegrationMethod | None,
    tolerances: ErrorTolerances,
    density_coordinates: DensityCoordinates | None = None,
    electron_ndof: int | None = None,
) -> DensityProblem:
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
    integration = resolve_integration(
        hamiltonian,
        kT=kT,
        integration=integration,
        superconducting=electron_ndof is not None,
    )
    tolerances = resolve_integration_tolerances(integration, tolerances)
    size = tb_orbital_count(hamiltonian)
    if density_coordinates is None:
        coordinates = full_density_coordinates(
            normalize_keys(hamiltonian, keys), size=size
        )
    else:
        normalize_keys(hamiltonian, list(density_coordinates.keys))
        coordinates = density_coordinates
    if coordinates.size != size:
        raise ValueError(
            "density coordinate matrix size must match the Hamiltonian shape"
        )
    layout = None
    if isinstance(integration, UniformGrid) and isinstance(
        integration.matrix_function, RationalFOE
    ):
        weights = np.ones(size)
        if electron_ndof is not None:
            weights[electron_ndof:] = 0.0
        layout = SparseRationalLayout.build(
            density_coordinates=coordinates,
            trace_weights_diag=weights,
            include_all_diagonal=True,
        )
    return DensityProblem(
        hamiltonian,
        kT,
        integration,
        tolerances,
        coordinates,
        electron_ndof,
        layout,
    )


def resolve_integration(hamiltonian, *, kT, integration=None, superconducting=False):
    """Resolve defaults and physics/backend compatibility at one boundary."""
    if not math.isfinite(kT) or kT < 0:
        raise ValueError(
            "meanfi supports only finite non-negative temperatures (kT >= 0)"
        )
    if integration is not None and not isinstance(integration, IntegrationMethod):
        raise TypeError("integration must be an IntegrationMethod instance")
    sparse = prefers_sparse_storage(hamiltonian)
    finite = tb_dimension(hamiltonian) == 0
    if finite and integration is not None:
        if integration.nk is not None or integration.initial_nk is not None:
            warnings.warn(
                "Finite systems do not use nk or initial_nk; these settings are ignored.",
                UserWarning,
                stacklevel=3,
            )
            integration = replace(integration, nk=None, initial_nk=None)
    if integration is None:
        if sparse:
            raise ValueError(
                "Automatic sparse integration requires an explicit method; choose FermiSimplex() for dense zero-temperature evaluation, or UniformGrid with an appropriate matrix function."
            )

        if kT == 0:
            if superconducting and not finite:
                raise ValueError(
                    "Zero-temperature superconducting calculations require an explicit UniformGrid(nk=...) integration setting."
                )
            integration = UniformGrid() if superconducting else FermiSimplex()
        else:
            integration = UniformGrid()
    if isinstance(integration, FermiSimplex):
        if superconducting:
            raise ValueError("Superconducting density requires UniformGrid")
        if kT != 0:
            raise ValueError("FermiSimplex requires kT == 0")
        return integration
    if not isinstance(integration, UniformGrid):
        raise TypeError("integration must be an IntegrationMethod instance")
    if kT == 0 and integration.nk is None and not finite:
        raise ValueError("Zero-temperature UniformGrid requires explicit nk")
    method = integration.matrix_function
    if method is None:
        if sparse and ((integration.nk is None and not finite) or kT <= 0):
            raise ValueError(
                "Automatic sparse UniformGrid evaluation requires kT > 0 and prescribed nk; explicitly choose DirectDiagonalization() to permit dense batches."
            )
        method = RationalFOE() if sparse else DirectDiagonalization()
    if isinstance(method, RationalFOE):
        if integration.nk is None and not finite:
            raise ValueError(
                "UniformGrid RationalFOE requires prescribed nk; adaptive RationalFOE is unsupported"
            )
        if kT <= 0:
            raise ValueError("UniformGrid RationalFOE requires kT > 0")
        if not sparse:
            raise ValueError(
                "UniformGrid RationalFOE is supported only for sparse matrices"
            )
    elif not isinstance(method, DirectDiagonalization):
        raise TypeError(
            "UniformGrid.matrix_function must be DirectDiagonalization or RationalFOE"
        )
    return replace(integration, matrix_function=method)
