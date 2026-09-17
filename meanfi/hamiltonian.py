"""Hamiltonian representations on the normalized Brillouin zone."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from meanfi.tb.ops import _tb_type, add_tb
from meanfi.tb.transforms import tb_to_kfunc
from meanfi.tb.validate import freeze_tb, tb_dimension, tb_orbital_count


@dataclass(frozen=True, eq=False)
class BlochHamiltonian:
    """Dense callable Hamiltonian on ``[0, 2*pi]^ndim``.

    ``function(k)`` takes a vector of coordinates in radians and returns an
    ``(ndof, ndof)`` finite Hermitian matrix. Keep its captured parameters fixed
    during a calculation. Integrals use the normalized BZ measure; coordinate
    transformations do not insert a Jacobian. Only normal states are supported.
    """

    function: Callable[[np.ndarray], np.ndarray]
    ndim: int
    ndof: int

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError("function must be callable")
        for name in ("ndim", "ndof"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def __call__(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        if k.shape != (self.ndim,) or not np.all(np.isfinite(k)):
            raise ValueError("Momentum coordinates must match ndim")
        matrix = np.asarray(self.function(k), dtype=complex)
        if matrix.shape != (self.ndof, self.ndof):
            raise ValueError("Callable Hamiltonian matrix shape must match ndof")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Callable Hamiltonian must return finite matrices")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-8, rtol=0.0):
            raise ValueError("Callable Hamiltonian must return Hermitian matrices")
        return matrix


Hamiltonian = _tb_type | BlochHamiltonian


def hamiltonian_dimension(h: Hamiltonian) -> int:
    return h.ndim if isinstance(h, BlochHamiltonian) else tb_dimension(h)


def hamiltonian_size(h: Hamiltonian) -> int:
    return h.ndof if isinstance(h, BlochHamiltonian) else tb_orbital_count(h)


def add_correction(h: Hamiltonian, correction: _tb_type) -> Hamiltonian:
    """Add a Fourier correction without changing the bare Hamiltonian."""
    if not isinstance(h, BlochHamiltonian):
        return add_tb(h, correction)
    if not correction:
        return h
    correction = freeze_tb(correction)
    if tb_dimension(correction) != h.ndim or tb_orbital_count(correction) != h.ndof:
        raise ValueError("Correction must match the Hamiltonian dimension and size")
    evaluate_correction = tb_to_kfunc(correction)
    return BlochHamiltonian(lambda k: h(k) + evaluate_correction(k), h.ndim, h.ndof)
