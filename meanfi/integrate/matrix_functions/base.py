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
class ChebyshevFOE(BdGMatrixFunction):
    """Evaluate BdG density block-columns with a Chebyshev Fermi-operator expansion."""

    initial_order: int = 16
    max_order: int = 1024
    coefficient_oversampling: int = 4
    spectral_padding: float = 1e-8
    dn_dmu_rtol: float = 1e-1
    trace_estimator: Literal["exact", "hutchinson"] = "exact"
    trace_probes: int = 16
    trace_seed: int = 0

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
        if self.trace_estimator not in {"exact", "hutchinson"}:
            raise ValueError("trace_estimator must be 'exact' or 'hutchinson'")
        if self.trace_probes <= 0:
            raise ValueError("trace_probes must be positive")


@dataclass(frozen=True)
class RationalFOE(BdGMatrixFunction):
    """Evaluate BdG density block-columns with an Ozaki rational Fermi-operator expansion."""

    initial_poles: int = 4
    max_poles: int = 256
    dn_dmu_rtol: float = 1e-1
    trace_estimator: Literal["exact", "hutchinson"] = "exact"
    trace_probes: int = 16
    trace_seed: int = 0
    rational_scheme: Literal["ozaki", "minimax", "aaa"] = "ozaki"

    def __post_init__(self) -> None:
        if self.initial_poles <= 0:
            raise ValueError("initial_poles must be positive")
        if self.max_poles < 2 * self.initial_poles:
            raise ValueError("max_poles must be at least twice initial_poles")
        if self.dn_dmu_rtol < 0:
            raise ValueError("dn_dmu_rtol must be non-negative")
        if self.trace_estimator not in {"exact", "hutchinson"}:
            raise ValueError("trace_estimator must be 'exact' or 'hutchinson'")
        if self.trace_probes <= 0:
            raise ValueError("trace_probes must be positive")
        if self.rational_scheme not in {"ozaki", "minimax", "aaa"}:
            raise ValueError("rational_scheme must be 'ozaki', 'minimax', or 'aaa'")


@dataclass(frozen=True)
class _BlockResult:
    block: np.ndarray
    derivative_block: np.ndarray | None
    error: float
    order: int | None


WorkspacePrecision = Literal[64, 128]
