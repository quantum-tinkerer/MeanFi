"""Local normal-ordered products of Hermitian fermion bilinears."""

from dataclasses import dataclass

import numpy as np

from meanfi.space.coordinates import DensityCoordinates, onsite_key
from meanfi.tb.ops import to_dense


@dataclass(frozen=True, eq=False)
class BilinearTerm:
    """One term ``coefficient : (c† A c)(c† B c) :``.

    ``A`` and ``B`` are Hermitian matrices in the complete orbital space. The
    coefficient is real, with no implicit factor of one half. Operators are
    copied into read-only storage. Normal ordering excludes one-body terms.
    """

    coefficient: float
    A: np.ndarray
    B: np.ndarray

    def __post_init__(self):
        coefficient = complex(self.coefficient)
        if not np.isfinite(coefficient) or coefficient.imag != 0:
            raise ValueError("Bilinear coefficient must be finite and real")
        object.__setattr__(self, "coefficient", coefficient.real)
        for name in ("A", "B"):
            operator = np.array(getattr(self, name), dtype=complex, copy=True)
            if (
                operator.ndim != 2
                or operator.shape[0] == 0
                or operator.shape[0] != operator.shape[1]
                or not np.all(np.isfinite(operator))
                or not np.allclose(operator, operator.conj().T, atol=1e-8, rtol=0.0)
            ):
                raise ValueError("Bilinear operators must be finite Hermitian matrices")
            operator.setflags(write=False)
            object.__setattr__(self, name, operator)
        if self.A.shape != self.B.shape:
            raise ValueError("Bilinear operators must have the same shape")


@dataclass(frozen=True, eq=False)
class BilinearInteraction:
    """Sum of local bilinear terms, for normal-state Hartree–Fock.

    All terms act in the same full orbital space. For onsite density ``rho``,
    each term has energy ``g * (Tr(A rho) Tr(B rho) - Tr(A rho B rho))``.
    Pairing and nonlocal bilinear products are not supported.
    """

    terms: tuple[BilinearTerm, ...]

    def __post_init__(self):
        terms = tuple(self.terms)
        if not terms or any(not isinstance(term, BilinearTerm) for term in terms):
            raise ValueError("terms must contain at least one BilinearTerm")
        if any(term.A.shape != terms[0].A.shape for term in terms):
            raise ValueError("All bilinear terms must have the same matrix size")
        object.__setattr__(self, "terms", terms)

    @property
    def ndof(self) -> int:
        return self.terms[0].A.shape[0]

    def density_coordinates(self, ndim: int) -> DensityCoordinates:
        """Select orbital blocks touched by each nonzero interaction term."""
        local = onsite_key(ndim)
        entries = set()
        for term in self.terms:
            if term.coefficient == 0:
                continue
            orbitals = np.flatnonzero(np.any((term.A != 0) | (term.B != 0), axis=0))
            entries.update((local, int(i), int(j)) for i in orbitals for j in orbitals)
        return DensityCoordinates.from_entries(
            size=self.ndof, keys=[local], entries=entries
        )

    def correction(self, rho: np.ndarray) -> np.ndarray:
        """Derivative of the Wick energy with respect to onsite density."""
        rho = to_dense(rho)
        result = np.zeros((self.ndof, self.ndof), dtype=complex)
        for term in self.terms:
            a, b, g = term.A, term.B, term.coefficient
            result += g * (
                np.trace(b @ rho) * a
                + np.trace(a @ rho) * b
                - a @ rho @ b
                - b @ rho @ a
            )
        return result
