"""Dense Fourier views of dense or sparse tight-binding blocks."""

from itertools import product
from numbers import Integral
from typing import Callable

import numpy as np
from scipy import sparse

from meanfi.tb.ops import _tb_type
from meanfi.tb.validate import tb_dimension, tb_orbital_count, validate_tb_dict


def tb_to_kgrid(tb: _tb_type, shape: tuple[int, ...]) -> np.ndarray:
    """Sample H(k) on an FFT-ordered grid with explicit points per axis.

    Axis n uses ``2*pi*numpy.fft.fftfreq(n)``. The returned dense array has
    shape ``(*shape, orbitals, orbitals)``; use ``shape=()`` for a finite system.
    Hoppings outside the sampled frequency range alias onto this grid.
    """
    validate_tb_dict(tb)
    shape = tuple(shape)
    if len(shape) != tb_dimension(tb) or any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in shape
    ):
        raise ValueError(
            "shape must contain one positive integer per lattice dimension"
        )
    size = tb_orbital_count(tb)
    coefficients = np.zeros((*shape, size, size), dtype=complex)
    for key, matrix in tb.items():
        index = tuple(r % n for r, n in zip(key, shape, strict=True))
        coefficients[index] += matrix.toarray() if sparse.issparse(matrix) else matrix
    return (
        np.fft.fftn(coefficients, axes=tuple(range(len(shape))))
        if shape
        else coefficients
    )


def kgrid_to_tb(kgrid_array: np.ndarray) -> _tb_type:
    """Invert an FFT-ordered grid, preserving every sampled Fourier mode.

    Even axes share their Nyquist coefficient equally between +/-n/2. This
    preserves Hermiticity of physical inputs and exact grid round trips.
    """
    array = np.asarray(kgrid_array)
    axes = tuple(range(array.ndim - 2))
    return ifftn_to_tb(np.fft.ifftn(array, axes=axes) if axes else array)


def ifftn_to_tb(ifft_array: np.ndarray) -> _tb_type:
    """Convert inverse-FFT coefficients to blocks, splitting even-axis Nyquist modes."""
    array = np.asarray(ifft_array)
    if array.ndim < 2 or array.shape[-2] != array.shape[-1] or 0 in array.shape:
        raise ValueError("expected nonempty grid axes followed by square matrix blocks")
    shape = array.shape[:-2]
    result = {}
    for key in product(*(range(-(n // 2), n // 2 + 1) for n in shape)):
        nyquist_axes = sum(
            n % 2 == 0 and abs(r) == n // 2 for r, n in zip(key, shape, strict=True)
        )
        result[key] = array[key].copy() / 2**nyquist_axes
    return result


def tb_to_kfunc(tb: _tb_type) -> Callable:
    """Return H(k) for one point or arrays with final axis ``dimension``.

    k is in radians. Outputs are dense; sparse inputs retain sparse storage
    until their nonzero entries are added to the result.
    """
    validate_tb_dict(tb)
    dimension, size = tb_dimension(tb), tb_orbital_count(tb)
    dense_keys, dense_matrices, sparse_blocks = [], [], []
    for key, matrix in tb.items():
        if sparse.issparse(matrix):
            matrix = matrix.tocoo(copy=True)
            matrix.sum_duplicates()
            sparse_blocks.append((np.asarray(key), matrix))
        else:
            dense_keys.append(key)
            dense_matrices.append(np.asarray(matrix, dtype=complex))
    if dense_keys:
        keys = np.asarray(dense_keys)
        matrices = np.stack(dense_matrices)

    def kfunc(k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        if k.ndim == 0 or k.shape[-1] != dimension:
            raise ValueError("k must have a final axis matching the lattice dimension")
        if dense_keys:
            phases = np.exp(-1j * (k @ keys.T))
            result = np.tensordot(phases, matrices, axes=(-1, 0))
        else:
            result = np.zeros((*k.shape[:-1], size, size), dtype=complex)
        for key, matrix in sparse_blocks:
            phase = np.exp(-1j * (k @ key))
            result[..., matrix.row, matrix.col] += phase[..., None] * matrix.data
        return result

    return kfunc
