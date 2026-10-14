"""Hamiltonian representations on the normalized Brillouin zone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, KW_ONLY
from functools import wraps
from inspect import Parameter, signature

import numpy as np

from meanfi.tb.ops import _tb_type, add_tb, block_diag, to_dense
from meanfi.tb.bdg import electron_to_bdg_tb
from meanfi.tb.transforms import tb_to_kfunc
from meanfi.tb.validate import freeze_tb, tb_dimension, tb_orbital_count


def _square_matrix(value, *, ndof=None) -> np.ndarray:
    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Callable Hamiltonian must return a nonempty square matrix")
    if ndof is not None and matrix.shape != (ndof, ndof):
        raise ValueError("Callable Hamiltonian matrix shape must remain constant")
    return matrix


@dataclass(frozen=True, eq=False)
class BlochHamiltonian:
    """Dense callable Hamiltonian on the normalized Brillouin zone.

    ``function(kx, ky, ...)`` takes separate coordinates in radians. Dimension
    is inferred from its required positional arguments; the orbital count is
    inferred from its matrix at the origin. The function must have an inspectable
    signature with only required positional coordinates. Bind physical parameters
    in a closure or partial before constructing the Hamiltonian.

    Every evaluation must return a finite Hermitian matrix of the same size.
    The caller is responsible for finite coordinates and finite Hermitian
    outputs; evaluations check shapes without scanning matrix entries.
    Keep captured parameters fixed during a calculation. Integration uses
    ``[0, 2*pi]^ndim`` with normalized measure; coordinate transformations do not
    insert a Jacobian automatically. ``transform`` carries a constant physical
    measure explicitly. BdG uses the partner momentum ``-k mod 2*pi``; a domain
    wrapper must preserve this relation to physical momentum reversal.
    """

    function: Callable[..., np.ndarray]
    _: KW_ONLY
    measure: float = 1.0
    ndim: int = field(init=False)
    ndof: int = field(init=False)

    def __post_init__(self):
        measure = float(self.measure)
        if not np.isfinite(measure) or measure <= 0:
            raise ValueError("measure must be finite and positive")
        object.__setattr__(self, "measure", measure)
        if not callable(self.function):
            raise TypeError("function must be callable")
        try:
            parameters = tuple(signature(self.function).parameters.values())
        except (TypeError, ValueError) as error:
            raise TypeError("function must have an inspectable signature") from error
        if not parameters or any(
            parameter.kind
            not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            or parameter.default is not Parameter.empty
            for parameter in parameters
        ):
            raise TypeError(
                "function must accept only required positional momentum coordinates"
            )
        ndim = len(parameters)
        matrix = _square_matrix(self.function(*np.zeros(ndim)))
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "ndof", matrix.shape[0])

    def __call__(self, *k) -> np.ndarray:
        coordinates = np.asarray(k, dtype=float)
        if coordinates.shape != (self.ndim,):
            raise ValueError("Momentum coordinates must match ndim")
        return _square_matrix(self.function(*coordinates), ndof=self.ndof)

    def transform(self, coordinates: Callable, *, measure: float) -> BlochHamiltonian:
        """Pull back a Hamiltonian through an equal-measure coordinate map.

        ``coordinates(*k)`` returns the arguments of this Hamiltonian. ``measure``
        is the total measure relative to the normalized computational domain,
        including physical unit conversions. Composition multiplies measures.
        Only a constant measure is supported; variable Jacobians are not.
        Densities, filling, energies and absolute errors carry the resulting
        measure. Hamiltonian values, temperature and chemical potential do not.
        """
        if not callable(coordinates):
            raise TypeError("coordinates must be callable")

        @wraps(coordinates)
        def transformed(*k):
            return self(*coordinates(*k))

        return BlochHamiltonian(transformed, measure=self.measure * measure)


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
    local = (0,) * h.ndim
    onsite = to_dense(correction[local]) if set(correction) == {local} else None
    evaluate_correction = None if onsite is not None else tb_to_kfunc(correction)

    @wraps(h.function)
    def corrected(*k):
        # Check shape before addition so broadcasting cannot conceal invalid input.
        matrix = _square_matrix(h.function(*k), ndof=h.ndof)
        return matrix + (
            onsite if onsite is not None else evaluate_correction(np.asarray(k))
        )

    return BlochHamiltonian(corrected, measure=h.measure)


def electron_to_bdg(h: Hamiltonian) -> Hamiltonian:
    """Embed an electron Hamiltonian using the standard BZ momentum reversal."""
    if not isinstance(h, BlochHamiltonian):
        return electron_to_bdg_tb(h, tb_orbital_count(h))

    @wraps(h.function)
    def bdg(*k):
        opposite = np.mod(-np.asarray(k), 2 * np.pi)
        return block_diag(h(*k), -h(*opposite).T)

    return BlochHamiltonian(bdg, measure=h.measure)
