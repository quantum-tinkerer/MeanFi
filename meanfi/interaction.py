"""Finite-range normal-ordered products of Hermitian fermion bilinears."""

from dataclasses import dataclass, KW_ONLY
from numbers import Integral

import numpy as np

from meanfi.space.coordinates import (
    DensityCoordinates,
    canonical_tb_keys,
    onsite_key,
    opposite_key,
)
from meanfi.tb.bdg import assemble_bdg_tb
from meanfi.tb.ops import _tb_type, to_dense
from meanfi.tb.validate import tb_dimension


@dataclass(frozen=True, eq=False)
class BilinearTerm:
    """Term ``g sum_x : (c_x† A c_x)(c_{x+R}† B c_{x+R}) :``.

    ``A`` and ``B`` are Hermitian matrices in the complete orbital space.
    ``displacement`` gives R as an integer tuple; None means onsite in the
    model's dimension. List each term once: both correction directions are
    generated automatically. ``(A, B, R)`` and ``(B, A, -R)`` describe the same
    interaction. Listing both at full weight doubles it; terms are additive and
    are not deduplicated. Reversing R without swapping A and B generally gives
    a different interaction. There is no implicit factor of one half in g.
    Operators are copied into read-only storage. Normal ordering excludes
    one-body terms.
    """

    coefficient: float
    A: np.ndarray
    B: np.ndarray
    _: KW_ONLY
    displacement: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.displacement is not None:
            displacement = tuple(self.displacement)
            if any(
                isinstance(r, bool) or not isinstance(r, Integral) for r in displacement
            ):
                raise ValueError("displacement must contain integers")
            object.__setattr__(self, "displacement", displacement)
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
    """Finite-range bilinear interactions for normal or BdG mean-field states.

    All terms act in the same full orbital space. The normal energy of a term
    at displacement R is ``g * (Tr(A rho_0) Tr(B rho_0) - Tr(A rho_R B rho_-R))``.
    BdG adds ``g * Tr(kappa_R† A kappa_R B.T)``. Reference subtraction and energy
    normalization are handled by Model, as for density-density interactions.
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

    def _terms_with_keys(self, ndim):
        local = onsite_key(ndim)
        for term in self.terms:
            key = local if term.displacement is None else term.displacement
            if len(key) != ndim:
                raise ValueError(
                    "Bilinear displacement must match the Hamiltonian dimension"
                )
            yield term, key, opposite_key(key)

    def density_coordinates(
        self, ndim: int, *, superconducting=False
    ) -> DensityCoordinates:
        """Select Hartree, exchange and pairing entries at the required displacements."""
        local = onsite_key(ndim)
        entries, keys = set(), {local}
        for term, key, opposite in self._terms_with_keys(ndim):
            keys.update((key, opposite))
            if term.coefficient == 0:
                continue
            for operator in (term.A, term.B):
                rows, cols = np.nonzero(operator)
                entries.update(
                    (local, int(i), int(j)) for i, j in zip(rows, cols, strict=True)
                )
            a_orbitals = np.flatnonzero(np.any(term.A != 0, axis=0))
            b_orbitals = np.flatnonzero(np.any(term.B != 0, axis=0))
            for i in a_orbitals:
                for j in b_orbitals:
                    entries.update(((key, int(i), int(j)), (opposite, int(j), int(i))))
                    if superconducting:
                        entries.update(
                            (
                                (key, int(i), self.ndof + int(j)),
                                (opposite, int(j), self.ndof + int(i)),
                            )
                        )
        return DensityCoordinates.from_entries(
            size=(2 if superconducting else 1) * self.ndof,
            keys=canonical_tb_keys(keys),
            entries=entries,
        )

    def correction(self, density: _tb_type, *, superconducting=False) -> _tb_type:
        """Apply the linear Wick map to validated normal or Nambu density blocks."""
        ndim, size = tb_dimension(density), self.ndof
        local = onsite_key(ndim)
        normal = {key: np.zeros((size, size), complex) for key in density}
        pairing = (
            {key: np.zeros((size, size), complex) for key in density}
            if superconducting
            else None
        )
        blocks = {key: to_dense(block) for key, block in density.items()}
        rho_0 = blocks[local][:size, :size]
        for term, key, opposite in self._terms_with_keys(ndim):
            a, b, g = term.A, term.B, term.coefficient
            normal[local] += g * (np.trace(b @ rho_0) * a + np.trace(a @ rho_0) * b)
            normal[key] -= g * a @ blocks[key][:size, :size] @ b
            normal[opposite] -= g * b @ blocks[opposite][:size, :size] @ a
            if superconducting:
                pairing[key] += g * a @ blocks[key][:size, size:] @ b.T
                pairing[opposite] += g * b @ blocks[opposite][:size, size:] @ a.T
        return (
            assemble_bdg_tb(normal, pairing, ndof=size) if superconducting else normal
        )
