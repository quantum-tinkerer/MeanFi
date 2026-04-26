from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SCFMethod:
    """Base class for SCF fixed-point solvers."""

    max_iterations: int = 100

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class LinearMixing(SCFMethod):
    """Simple linear mixing on the reduced density-matrix parameters."""

    alpha: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")


@dataclass(frozen=True)
class AndersonMixing(SCFMethod):
    """Anderson acceleration via SciPy's fixed-point root solver."""

    M: int = 0
    line_search: str = "wolfe"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.M < 0:
            raise ValueError("M must be non-negative")
