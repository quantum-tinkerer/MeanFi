from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BdGMatrixFunction",
    "ExactDiagonalization",
    "ChebyshevFOE",
]


@dataclass(frozen=True)
class BdGMatrixFunction:
    """Base class for BdG matrix-function density evaluators."""


@dataclass(frozen=True)
class ExactDiagonalization(BdGMatrixFunction):
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
