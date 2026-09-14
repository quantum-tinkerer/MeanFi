from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DirectDiagonalization:
    """Evaluate normal or BdG density by direct diagonalization."""


@dataclass(frozen=True)
class RationalFOE:
    """Evaluate selected sparse density entries with a rational Fermi expansion."""

    initial_poles: int = 4
    max_poles: int = 256
    rational_scheme: Literal["ozaki", "aaa"] = "aaa"

    def __post_init__(self) -> None:
        if self.initial_poles <= 0:
            raise ValueError("initial_poles must be positive")
        if self.max_poles < 2 * self.initial_poles:
            raise ValueError("max_poles must be at least twice initial_poles")
        if self.rational_scheme not in {"ozaki", "aaa"}:
            raise ValueError("rational_scheme must be 'ozaki' or 'aaa'")
