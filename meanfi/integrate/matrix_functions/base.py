from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BdGMatrixFunction:
    """Base class for BdG matrix-function density evaluators."""


@dataclass(frozen=True)
class DirectDiagonalization(BdGMatrixFunction):
    """Evaluate the finite-temperature BdG density matrix by diagonalization."""


@dataclass(frozen=True)
class ChebyshevFOE(BdGMatrixFunction):
    """Evaluate BdG density block-columns with a Chebyshev Fermi-operator expansion."""

    initial_order: int = 16
    max_order: int = 1024
    coefficient_oversampling: int = 4
    spectral_padding: float = 1e-8
    dn_dmu_rtol: float = 1e-1

    def __post_init__(self) -> None:
        if self.initial_order <= 0:
            raise ValueError("initial_order must be positive")
        if self.max_order < 2 * self.initial_order:
            raise ValueError("max_order must be at least twice initial_order")
        if self.coefficient_oversampling <= 0:
            raise ValueError("coefficient_oversampling must be positive")
        if self.spectral_padding < 0:
            raise ValueError("spectral_padding must be non-negative")
        if self.dn_dmu_rtol < 0:
            raise ValueError("dn_dmu_rtol must be non-negative")


@dataclass(frozen=True)
class RationalFOE(BdGMatrixFunction):
    """Evaluate BdG density block-columns with an Ozaki rational Fermi-operator expansion."""

    initial_poles: int = 4
    max_poles: int = 256
    dn_dmu_rtol: float = 1e-1

    def __post_init__(self) -> None:
        if self.initial_poles <= 0:
            raise ValueError("initial_poles must be positive")
        if self.max_poles < 2 * self.initial_poles:
            raise ValueError("max_poles must be at least twice initial_poles")
        if self.dn_dmu_rtol < 0:
            raise ValueError("dn_dmu_rtol must be non-negative")


@dataclass(frozen=True)
class _BlockResult:
    block: np.ndarray
    derivative_block: np.ndarray | None
    error: float
    order: int | None
