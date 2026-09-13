from __future__ import annotations

import numpy as np

from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like, matrix_shape


def matrix_allclose(lhs, rhs, *, atol: float = 1e-8) -> bool:
    """Compare matrix entries without materializing sparse inputs."""

    if matrix_shape(lhs) != matrix_shape(rhs):
        return False
    if is_sparse_like(lhs) or is_sparse_like(rhs):
        difference = as_sparse(lhs) - as_sparse(rhs)
        return bool(np.all(np.abs(difference.data) <= atol))
    return bool(np.allclose(lhs, rhs, atol=atol, rtol=0.0))


def tb_dimension(tb: _tb_type) -> int:
    try:
        return len(next(iter(tb)))
    except StopIteration as exc:
        raise ValueError("Tight-binding dictionaries must be non-empty") from exc


def tb_orbital_count(tb: _tb_type) -> int:
    rows, cols = matrix_shape(next(iter(tb.values())))
    if rows != cols:
        raise ValueError("Tight-binding values must be square matrices")
    return rows


def zero_key(ndim: int) -> tuple[int, ...]:
    return (0,) * ndim


def validate_tb_dict(tb: _tb_type) -> None:
    ndim = tb_dimension(tb)
    n_orbitals = tb_orbital_count(tb)

    for key, value in tb.items():
        if len(key) != ndim:
            raise ValueError("All hopping keys need to have the same length")
        if matrix_shape(value) != (n_orbitals, n_orbitals):
            raise ValueError("All hopping matrices need to have the same shape")


def validate_hermiticity(tb: _tb_type) -> None:
    for key, value in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        partner = tb.get(opposite)
        if partner is None:
            partner = value * 0
        if not matrix_allclose(value, partner.conj().T):
            raise ValueError("The provided tight-binding model is not hermitian.")


def normalize_keys(
    hamiltonian: _tb_type,
    keys: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    normalized = [tuple(int(component) for component in key) for key in keys]
    ndim = tb_dimension(hamiltonian)
    for key in normalized:
        if len(key) != ndim:
            raise ValueError(
                "Requested density-matrix keys must match the Hamiltonian dimension"
            )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Requested density-matrix keys must be unique")
    return normalized


def require_zero_dim_local_key_only(hamiltonian: _tb_type) -> None:
    if tb_dimension(hamiltonian) != 0:
        raise ValueError("This helper expects a zero-dimensional tight-binding input")
    if tuple() not in hamiltonian or len(hamiltonian) != 1:
        raise ValueError(
            "Zero-dimensional Hamiltonians must contain only the local key"
        )
