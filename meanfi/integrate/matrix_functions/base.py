from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BdGMatrixFunction:
    """Base class for BdG matrix-function density evaluators."""


@dataclass(frozen=True)
class DirectDiagonalization(BdGMatrixFunction):
    """Evaluate the finite-temperature BdG density matrix by diagonalization."""


@dataclass(frozen=True)
class RationalFOE(BdGMatrixFunction):
    """Evaluate BdG density block-columns with a rational Fermi-operator expansion."""

    initial_poles: int = 4
    max_poles: int = 256
    dn_dmu_rtol: float = 1e-1
    rational_scheme: Literal["ozaki", "aaa"] = "ozaki"

    def __post_init__(self) -> None:
        if self.initial_poles <= 0:
            raise ValueError("initial_poles must be positive")
        if self.max_poles < 2 * self.initial_poles:
            raise ValueError("max_poles must be at least twice initial_poles")
        if self.dn_dmu_rtol < 0:
            raise ValueError("dn_dmu_rtol must be non-negative")
        if self.rational_scheme not in {"ozaki", "aaa"}:
            raise ValueError("rational_scheme must be 'ozaki' or 'aaa'")


@dataclass(frozen=True)
class _BlockResult:
    block: np.ndarray
    derivative_block: np.ndarray | None
    error: float
    order: int | None


WorkspacePrecision = Literal[64, 128]
