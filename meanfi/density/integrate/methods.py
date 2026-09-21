from __future__ import annotations

from dataclasses import dataclass


FILLING_TOLERANCE_ESTIMATOR_FACTOR = 5.4


@dataclass(frozen=True)
class IntegrationMethod:
    """Base class for Brillouin-zone integration strategies."""


@dataclass(frozen=True)
class AdaptiveSimplex(IntegrationMethod):
    """Charge mesh refinement followed by density p-cubature at zero temperature.

    ``density_max_degree`` caps density cubature (2 or odd, up to 21).
    ``max_refinements`` limits charge splits and density order promotions
    separately. Density cubature holds cut-band occupation fractions fixed.
    """

    density_matrix_tol: float | None = None
    max_refinements: int | None = None
    num_threads: int | None = 1
    charge_tol: float | None = None
    density_max_degree: int = 21

    def __post_init__(self) -> None:
        if not isinstance(self.density_max_degree, int) or (
            self.density_max_degree != 2
            and (
                self.density_max_degree < 3
                or self.density_max_degree > 21
                or self.density_max_degree % 2 == 0
            )
        ):
            raise ValueError(
                "density_max_degree must be 2 or an odd integer in [3, 21]"
            )
        if self.density_matrix_tol is not None and self.density_matrix_tol <= 0:
            raise ValueError("density_matrix_tol must be positive when provided")
        if self.charge_tol is not None and self.charge_tol <= 0:
            raise ValueError("charge_tol must be positive when provided")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative or None")
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive or None")


@dataclass(frozen=True)
class AdaptiveQuadrature(IntegrationMethod):
    """Adaptive finite-temperature quadrature."""

    density_matrix_tol: float | None = None
    max_refinements: int | None = None
    rule: str = "auto"
    batch_size: int | None = None
    matrix_function: object | None = None
    workspace_precision: int = 128
    charge_tol: float | None = None

    def __post_init__(self) -> None:
        if self.density_matrix_tol is not None and self.density_matrix_tol <= 0:
            raise ValueError("density_matrix_tol must be positive when provided")
        if self.charge_tol is not None and self.charge_tol <= 0:
            raise ValueError("charge_tol must be positive when provided")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative or None")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")
        if self.workspace_precision not in (64, 128):
            raise ValueError("workspace_precision must be 64 or 128")


@dataclass(frozen=True)
class UniformGrid(IntegrationMethod):
    """Uniform k-grid point sampling."""

    nk: int
    density_matrix_tol: float | None = None
    matrix_function: object | None = None
    workspace_precision: int = 128
    charge_tol: float | None = None

    def __post_init__(self) -> None:
        if self.nk <= 0:
            raise ValueError("nk must be positive")
        if self.density_matrix_tol is not None and self.density_matrix_tol <= 0:
            raise ValueError("density_matrix_tol must be positive when provided")
        if self.charge_tol is not None and self.charge_tol <= 0:
            raise ValueError("charge_tol must be positive when provided")
        if self.workspace_precision not in (64, 128):
            raise ValueError("workspace_precision must be 64 or 128")
