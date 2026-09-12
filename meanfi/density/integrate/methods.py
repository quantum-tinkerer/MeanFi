"""The two public Brillouin-zone integration families.

An explicit ``nk`` selects a prescribed mesh; otherwise tolerances control
refinement. Resource limits never select a mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import math


def _positive_integer(name, value, *, allow_none=False, minimum=1):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
            + (" or None" if allow_none else "")
        )


@dataclass(frozen=True)
class IntegrationMethod:
    """Base class for Brillouin-zone integration strategies."""


def _validate_mesh_settings(method):
    _positive_integer("nk", method.nk, allow_none=True)
    for name in ("density_matrix_tol", "charge_tol"):
        value = getattr(method, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite when provided")
    if method.nk is not None and (
        method.density_matrix_tol is not None or method.charge_tol is not None
    ):
        raise ValueError(
            "nk cannot be combined with explicit integration accuracy targets"
        )
    _positive_integer("max_points", method.max_points)
    _positive_integer(
        "max_refinements", method.max_refinements, allow_none=True, minimum=0
    )


@dataclass(frozen=True)
class AdaptiveSimplex(IntegrationMethod):
    """FermiSimplex integration of normal systems at zero temperature.

    ``nk`` requests a total number of native mesh vertices, including distinct
    boundary vertices. Native dyadic construction may overshoot this request.
    Without ``nk``, use empirical integration targets and adaptive refinement.
    """

    density_matrix_tol: float | None = None
    max_refinements: int | None = None
    num_threads: int | None = 1
    charge_tol: float | None = None
    nk: int | None = None
    max_points: int = 1_048_576

    def __post_init__(self):
        _validate_mesh_settings(self)
        _positive_integer("num_threads", self.num_threads, allow_none=True)


@dataclass(frozen=True)
class PeriodicGrid(IntegrationMethod):
    """Isotropic periodic point sampling with optional global refinement.

    ``nk`` requests TOTAL mesh points, rounded up to ``n**dimension``. Without
    ``nk``, finite-temperature integration doubles each axis and validates
    convergence with a shifted grid. ``batch_size`` bounds transient matrix
    storage; ``max_spectrum_bytes`` bounds retained normal-state eigenvalues.
    """

    nk: int | None = None
    density_matrix_tol: float | None = None
    charge_tol: float | None = None
    max_points: int = 1_048_576
    max_refinements: int | None = 12
    batch_size: int = 128
    max_spectrum_bytes: int = 256 * 1024 * 1024
    matrix_function: object | None = None
    workspace_precision: int = 128

    def __post_init__(self):
        _validate_mesh_settings(self)
        _positive_integer("batch_size", self.batch_size)
        _positive_integer("max_spectrum_bytes", self.max_spectrum_bytes)
        if self.workspace_precision not in (64, 128):
            raise ValueError("workspace_precision must be 64 or 128")
