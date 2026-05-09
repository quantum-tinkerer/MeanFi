from __future__ import annotations

import numpy as np

from meanfi.tb.ops import _tb_type, to_dense


def matrix_array(value) -> np.ndarray:
    return to_dense(value)


def matrix_allclose(lhs, rhs, *, atol: float = 1e-8) -> bool:
    return np.allclose(matrix_array(lhs), matrix_array(rhs), atol=atol, rtol=0.0)


def tb_dimension(tb: _tb_type) -> int:
    try:
        return len(next(iter(tb)))
    except StopIteration as exc:
        raise ValueError("Tight-binding dictionaries must be non-empty") from exc


def tb_orbital_count(tb: _tb_type) -> int:
    matrix = matrix_array(next(iter(tb.values())))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Tight-binding values must be square matrices")
    return int(matrix.shape[0])


def zero_key(ndim: int) -> tuple[int, ...]:
    return tuple(np.zeros((ndim,), dtype=int))


def validate_tb_dict(tb: _tb_type) -> None:
    ndim = tb_dimension(tb)
    n_orbitals = tb_orbital_count(tb)

    for key, value in tb.items():
        if len(key) != ndim:
            raise ValueError("All hopping keys need to have the same length")
        matrix = matrix_array(value)
        if matrix.shape != (n_orbitals, n_orbitals):
            raise ValueError("All hopping matrices need to have the same shape")


def validate_bdg_state(tb: _tb_type, *, ndof: int, name: str = "BdG correction") -> None:
    from meanfi.tb.bdg import validate_bdg_tb

    validate_bdg_tb(tb, ndof=ndof, ndim=tb_dimension(tb), name=name)


def validate_hermiticity(tb: _tb_type) -> None:
    matrix_shape = next(iter(tb.values())).shape
    zero = np.zeros(matrix_shape, dtype=np.complex128)
    for key, value in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        if not matrix_allclose(value, np.conj(tb.get(opposite, zero).T)):
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
