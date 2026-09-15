from __future__ import annotations

import numpy as np

from meanfi.space.coordinates import DensityCoordinates


def density_values_from_eigensystem(
    eigenvectors: np.ndarray,
    occupation: np.ndarray,
    coords: DensityCoordinates,
    *,
    phases: np.ndarray | None = None,
) -> np.ndarray:
    """Compute selected density values directly from eigenvectors.

    The returned vector follows ``coords.iter_key_coordinates()`` order and never forms
    selected columns as an intermediate representation.
    """

    # Many requested entries are cheaper as a matrix product. Few entries
    # use row-wise contractions, avoiding a full density matrix.
    full = None
    if any(rows.size > coords.size for rows in coords.rows_by_key):
        full = (eigenvectors * occupation[..., None, :]) @ eigenvectors.conj().swapaxes(
            -1, -2
        )
    values = np.empty(
        np.shape(eigenvectors)[:-2] + (coords.value_count,),
        dtype=complex,
    )
    for group_index, (_key, rows, cols, value_slice) in enumerate(
        coords.iter_key_coordinates()
    ):
        selected = (
            full[..., rows, cols]
            if full is not None
            else np.einsum(
                "...pa,...a,...pa->...p",
                eigenvectors[..., rows, :],
                occupation,
                eigenvectors[..., cols, :].conj(),
                optimize=True,
            )
        )
        if phases is not None:
            selected = selected * phases[..., group_index, np.newaxis]
        values[..., value_slice] = selected
    return values
