from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, kw_only=True)
class DirectDiagonalization:
    """Evaluate normal or BdG density by direct diagonalization."""


@dataclass(frozen=True, kw_only=True)
class RationalFOE:
    """Evaluate sparse density and entropy with a shared AAA pole expansion."""

    initial_poles: int = 4
    max_poles: int = 256

    def __post_init__(self) -> None:
        for name in ("initial_poles", "max_poles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_poles < self.initial_poles:
            raise ValueError("max_poles must be at least initial_poles")
