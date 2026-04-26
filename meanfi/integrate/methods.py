from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationMethod:
    """Base class for Brillouin-zone integration strategies."""


@dataclass(frozen=True)
class AdaptiveSimplex(IntegrationMethod):
    """Adaptive zero-temperature simplicial integration."""

    density_matrix_tol: float = 1e-6
    max_refinements: int | None = None

    def __post_init__(self) -> None:
        if self.density_matrix_tol <= 0:
            raise ValueError("density_matrix_tol must be positive")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative or None")


@dataclass(frozen=True)
class AdaptiveQuadrature(IntegrationMethod):
    """Adaptive finite-temperature quadrature."""

    density_matrix_tol: float = 1e-6
    max_refinements: int | None = None
    rule: str = "auto"
    batch_size: int | None = None
    matrix_function: object | None = None

    def __post_init__(self) -> None:
        if self.density_matrix_tol <= 0:
            raise ValueError("density_matrix_tol must be positive")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative or None")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")


@dataclass(frozen=True)
class UniformGrid(IntegrationMethod):
    """Uniform zero-temperature k-grid point sampling."""

    nk: int

    def __post_init__(self) -> None:
        if self.nk <= 0:
            raise ValueError("nk must be positive")
