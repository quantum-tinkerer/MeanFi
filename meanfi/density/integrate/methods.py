"""The two public Brillouin-zone integration families.

An explicit ``nk`` selects a prescribed mesh; otherwise tolerances control
refinement. Resource limits never select a mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np
from meanfi.density.kpoint.matrix_functions import DirectDiagonalization, RationalFOE


def _positive_integer(name, value, *, allow_none=False, minimum=1):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
            + (" or None" if allow_none else "")
        )


@dataclass(frozen=True, kw_only=True)
class IntegrationMethod:
    """Base class for Brillouin-zone integration strategies."""


def _validate_mesh_settings(method):
    _positive_integer("nk", method.nk, allow_none=True)
    _positive_integer("initial_nk", method.initial_nk, allow_none=True)
    if method.nk is not None and method.initial_nk is not None:
        raise ValueError("nk and initial_nk are mutually exclusive")
    _positive_integer("max_points", method.max_points)
    _positive_integer(
        "max_refinements", method.max_refinements, allow_none=True, minimum=0
    )


@dataclass(frozen=True, kw_only=True)
class FermiSimplex(IntegrationMethod):
    """FermiSimplex integration of normal systems at zero temperature.

    ``nk`` requests a total number of native mesh vertices, including distinct
    boundary vertices. Native dyadic construction may overshoot this request.
    Without ``nk``, use integration targets and adaptive refinement.
    ``initial_nk`` sets the starting size; the default is 5**dimension vertices.
    Adaptive density uses p-cubature on the charge mesh, capped by
    ``density_max_degree``. Prescribed ``nk`` retains the existing density
    rule. Energy uses cached vertex eigenvalues and new centroid eigenvalues
    in a degree-two simplex rule; density targets do not bound its error.
    """

    max_refinements: int | None = None
    num_threads: int | None = 1
    nk: int | None = None
    initial_nk: int | None = None
    max_points: int = 1_048_576
    density_max_degree: int = 21

    def __post_init__(self):
        _validate_mesh_settings(self)
        _positive_integer("num_threads", self.num_threads, allow_none=True)
        degree = self.density_max_degree
        if (
            isinstance(degree, bool)
            or not isinstance(degree, Integral)
            or (degree != 2 and (degree < 3 or degree > 21 or degree % 2 == 0))
        ):
            raise ValueError(
                "density_max_degree must be 2 or an odd integer in [3, 21]"
            )


@dataclass(frozen=True, kw_only=True)
class UniformGrid(IntegrationMethod):
    """Isotropic periodic point sampling with optional global refinement.

    ``nk`` requests TOTAL mesh points, rounded up to ``n**dimension``. Without
    ``nk``, finite-temperature integration doubles each axis and validates
    convergence by comparing coarse and fine integrals. ``initial_nk`` sets
    the starting size; the default is 4**dimension points. ``batch_size`` bounds transient matrix
    storage; ``max_spectrum_bytes`` bounds retained normal-state eigenvalues.
    At zero temperature, only fixed-mu evaluation is supported; periodic systems
    also require explicit ``nk``. Fixed-filling root searches require kT > 0.
    """

    nk: int | None = None
    initial_nk: int | None = None
    max_points: int = 1_048_576
    max_refinements: int | None = 12
    batch_size: int = 128
    max_spectrum_bytes: int = 256 * 1024 * 1024
    matrix_function: DirectDiagonalization | RationalFOE | None = None
    dtype: str | np.dtype = "complex128"

    def __post_init__(self):
        _validate_mesh_settings(self)
        _positive_integer("batch_size", self.batch_size)
        _positive_integer("max_spectrum_bytes", self.max_spectrum_bytes)
        dtype = np.dtype(self.dtype)
        if dtype not in (np.dtype("complex64"), np.dtype("complex128")):
            raise ValueError("dtype must be complex64 or complex128")
        object.__setattr__(self, "dtype", dtype)
