from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from meanfi.tb.tb import _tb_type


def _matrix_array(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray(), dtype=complex)
    return np.asarray(value)


def tb_dimension(tb: _tb_type) -> int:
    """Return the spatial dimension encoded by a tight-binding dictionary."""

    return len(next(iter(tb)))


def tb_orbital_count(tb: _tb_type) -> int:
    """Return the number of internal degrees of freedom per unit cell."""

    return next(iter(tb.values())).shape[0]


def zero_key(ndim: int) -> tuple[int, ...]:
    """Return the onsite tight-binding key for a given dimension."""

    return (0,) * ndim


def validate_tb_dict(tb: _tb_type) -> None:
    """Validate the matrix shape consistency of a tight-binding dictionary."""

    size = None
    for value in tb.values():
        shape = getattr(value, "shape", None)
        if shape is None:
            raise ValueError(
                "Values of the tight-binding dictionary must be matrix-like"
            )

        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                "Values of the tight-binding dictionary must be square matrices"
            )

        if size is None:
            size = shape[0]
        elif shape[0] != size:
            raise ValueError(
                "Values of the tight-binding dictionary must have consistent shape"
            )


def validate_hermiticity(tb: _tb_type) -> None:
    """Validate that a tight-binding dictionary is Hermitian."""

    for vector, matrix in tb.items():
        opposite = tuple(-1 * np.asarray(vector))
        if not np.allclose(
            _matrix_array(matrix),
            _matrix_array(tb[opposite].conj().T),
        ):
            raise ValueError("Tight-binding dictionary must be hermitian.")


def normalize_keys(
    hamiltonian: _tb_type, keys: Iterable[Sequence[int]]
) -> list[tuple[int, ...]]:
    """Normalize requested density keys and validate their dimension."""

    normalized = [tuple(key) for key in keys]
    ndim = tb_dimension(hamiltonian)
    if any(len(key) != ndim for key in normalized):
        raise ValueError("All keys must have the same dimension as the Hamiltonian")
    return normalized


def require_zero_dim_local_key_only(hamiltonian: _tb_type) -> None:
    """Ensure zero-dimensional evaluation uses only the onsite matrix."""

    if set(hamiltonian) != {tuple()}:
        raise ValueError("Zero-dimensional evaluation expects only the local key")
