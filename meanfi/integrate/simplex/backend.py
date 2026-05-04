from __future__ import annotations

import numpy as np

from meanfi.tb.backend import tb_to_vertex_cache
from meanfi.tb.ops import _tb_type

try:
    from meanfi._zero_temp_ext import (
        AdaptiveIntegrator,
        ChargeSolveOptions,
        DensityIntegrateOptions,
        Geometry,
    )

    _ZERO_TEMP_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the extension is unavailable
    AdaptiveIntegrator = None
    ChargeSolveOptions = None
    DensityIntegrateOptions = None
    Geometry = None
    _ZERO_TEMP_EXT_AVAILABLE = False


_GEOM_TOL = 1e-14
_BULK_THETA = 0.5
_ROOT_SUBCELLS_PER_AXIS = 2
_UNLIMITED_MU_ITERATIONS = int(np.iinfo(np.int32).max)


def _root_subcells_per_axis(ndim: int) -> int:
    return 4 if ndim == 1 else _ROOT_SUBCELLS_PER_AXIS


def _preview_depth(refinement_depth: int) -> int:
    return int(refinement_depth) + 1


def _require_zero_temp_extension() -> None:
    if not _ZERO_TEMP_EXT_AVAILABLE or Geometry is None:
        raise RuntimeError(
            "Zero-temperature integration requires the compiled meanfi._zero_temp_ext extension"
        )


def _extension_subdivision_limit(max_subdivisions: int | None) -> int:
    return -1 if max_subdivisions is None else int(max_subdivisions)


def build_extension_runtime(hamiltonian: _tb_type, *, refinement_depth: int = 0):
    _require_zero_temp_extension()
    ndim = len(next(iter(hamiltonian)))
    geometry = Geometry.root(
        ndim,
        root_subcells_per_axis=_root_subcells_per_axis(ndim),
        tol=float(_GEOM_TOL),
    )
    vertex_cache = tb_to_vertex_cache(hamiltonian, tol=float(_GEOM_TOL))
    return geometry, vertex_cache, _preview_depth(refinement_depth)


def build_charge_options(
    *,
    mu_guess: float,
    charge_tol: float,
    mu_xtol: float,
    max_mu_iterations: int | None,
    max_subdivisions: int | None,
):
    options = ChargeSolveOptions()
    options.mu_guess = float(mu_guess)
    options.charge_tol = float(charge_tol)
    options.mu_xtol = float(mu_xtol)
    options.max_mu_iterations = (
        _UNLIMITED_MU_ITERATIONS if max_mu_iterations is None else int(max_mu_iterations)
    )
    options.max_subdivisions = _extension_subdivision_limit(max_subdivisions)
    options.bulk_theta = float(_BULK_THETA)
    return options


def build_density_options(
    *,
    density_atol: float,
    density_rtol: float,
    max_subdivisions: int | None,
    consumed_subdivisions: int = 0,
):
    options = DensityIntegrateOptions()
    options.density_atol = float(density_atol)
    options.density_rtol = float(density_rtol)
    if max_subdivisions is None:
        options.max_subdivisions = -1
    else:
        options.max_subdivisions = max(int(max_subdivisions) - int(consumed_subdivisions), 0)
    options.bulk_theta = float(_BULK_THETA)
    return options


def raise_normalized_runtime_error(exc: RuntimeError) -> None:
    if "Adaptive zero-temperature" in str(exc):
        raise ValueError(str(exc)) from exc
    raise exc
