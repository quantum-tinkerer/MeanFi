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

    alpha: float | None = None
    w0: float = 0.01
    M: int = 0
    f_rtol: float | None = None
    x_tol: float | None = None
    x_rtol: float | None = None
    line_search: str | None = "wolfe"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError("alpha must be positive when provided")
        if self.w0 < 0:
            raise ValueError("w0 must be non-negative")
        if self.M < 0:
            raise ValueError("M must be non-negative")
        if self.f_rtol is not None and self.f_rtol <= 0:
            raise ValueError("f_rtol must be positive when provided")
        if self.x_tol is not None and self.x_tol <= 0:
            raise ValueError("x_tol must be positive when provided")
        if self.x_rtol is not None and self.x_rtol <= 0:
            raise ValueError("x_rtol must be positive when provided")
        if self.line_search not in (None, "armijo", "wolfe"):
            raise ValueError("line_search must be None, 'armijo', or 'wolfe'")
