"""SCF settings; all controls are keyword-only."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral


def _positive_integer(name, value, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive(name, value):
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, kw_only=True)
class SCFMethod:
    max_iterations: int = 100

    def __post_init__(self):
        _positive_integer("max_iterations", self.max_iterations)


@dataclass(frozen=True, kw_only=True)
class LinearMixing(SCFMethod):
    """Mix a fraction ``alpha`` of the reduced density residual."""

    alpha: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        _positive("alpha", self.alpha)


@dataclass(frozen=True, kw_only=True)
class AndersonMixing(SCFMethod):
    """Anderson acceleration with a bounded initial mixing scale."""

    alpha: float = 0.5
    history_size: int = 5
    regularization: float = 0.01
    line_search: str | None = "armijo"

    def __post_init__(self):
        super().__post_init__()
        _positive("alpha", self.alpha)
        _positive_integer("history_size", self.history_size, minimum=0)
        if not isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and non-negative")
        if self.line_search not in (None, "armijo", "wolfe"):
            raise ValueError("line_search must be None, 'armijo', or 'wolfe'")


@dataclass(frozen=True, kw_only=True)
class EnergyDIIS(SCFMethod):
    """Minimize energy over a convex density history (normal T=0 simplex)."""

    history_size: int = 6

    def __post_init__(self):
        super().__post_init__()
        _positive_integer("history_size", self.history_size)
