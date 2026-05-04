from __future__ import annotations

import numpy as np

from meanfi.tb.ops import as_sparse, is_sparse_like, matrix_shape, sparse_module
from meanfi.tb.ops import _tb_type


def tb_k_matrix(hamiltonian: _tb_type, point: np.ndarray):
    accumulator = None
    for key, matrix in hamiltonian.items():
        phase = np.exp(-1j * np.dot(point, np.asarray(key, dtype=float)))
        term = matrix * phase
        accumulator = term if accumulator is None else accumulator + term
    if accumulator is None:
        raise ValueError("Hamiltonian cannot be empty")
    return accumulator


def build_tb_payload_helpers(hamiltonian: _tb_type):
    """Return a raw kernel payload and a payload-to-matrix reconstructor."""

    first_matrix = next(iter(hamiltonian.values()))
    shape = matrix_shape(first_matrix)
    if not any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
        size = shape[0]

        def kernel(points: np.ndarray) -> np.ndarray:
            payload = np.empty((points.shape[0], size * size), dtype=complex)
            for index, point in enumerate(points):
                payload[index] = np.asarray(
                    tb_k_matrix(hamiltonian, point),
                    dtype=complex,
                ).reshape(-1)
            return payload

        def matrix_from_payload(payload_row: np.ndarray):
            return np.asarray(payload_row, dtype=complex).reshape(shape)

        return kernel, matrix_from_payload

    sparse = sparse_module()
    structural = sparse.csr_matrix(shape, dtype=np.int8)
    term_payloads: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []
    for key, matrix in hamiltonian.items():
        csr_matrix = as_sparse(matrix).tocsr()
        coo_matrix = csr_matrix.tocoo()
        if coo_matrix.nnz > 0:
            structural = structural + sparse.csr_matrix(
                (
                    np.ones(coo_matrix.nnz, dtype=np.int8),
                    (coo_matrix.row, coo_matrix.col),
                ),
                shape=shape,
            )
        term_payloads.append(
            (
                key,
                np.stack([coo_matrix.row, coo_matrix.col], axis=1).astype(int, copy=False),
                np.array(coo_matrix.data, dtype=complex, copy=True),
            )
        )

    structural.sum_duplicates()
    structural = structural.tocsr()
    structural.sort_indices()
    structural.data[:] = 1

    locations: dict[tuple[int, int], int] = {}
    for row in range(shape[0]):
        start = int(structural.indptr[row])
        end = int(structural.indptr[row + 1])
        for offset in range(start, end):
            locations[(row, int(structural.indices[offset]))] = offset

    sparse_terms: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []
    for key, coordinates, data in term_payloads:
        if coordinates.size == 0:
            sparse_terms.append((key, np.empty(0, dtype=int), data))
            continue
        positions = np.fromiter(
            (locations[(int(row), int(col))] for row, col in coordinates),
            dtype=int,
            count=coordinates.shape[0],
        )
        sparse_terms.append((key, positions, data))

    def kernel(points: np.ndarray) -> np.ndarray:
        payload = np.zeros((points.shape[0], structural.nnz), dtype=complex)
        for key, positions, data in sparse_terms:
            if positions.size == 0:
                continue
            phase = np.exp(-1j * np.dot(points, np.asarray(key, dtype=float)))
            payload[:, positions] += phase[:, np.newaxis] * data[np.newaxis, :]
        return payload

    def matrix_from_payload(payload_row: np.ndarray):
        return sparse.csr_matrix(
            (
                np.asarray(payload_row, dtype=complex),
                structural.indices,
                structural.indptr,
            ),
            shape=shape,
        )

    return kernel, matrix_from_payload
