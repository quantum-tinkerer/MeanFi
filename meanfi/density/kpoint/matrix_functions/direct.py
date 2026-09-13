from __future__ import annotations

import numpy as np

from meanfi.space.coordinates import DensityCoordinates


def selected_density_values_from_eigensystem(
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

    values = np.empty(
        np.shape(eigenvectors)[:-2] + (coords.value_count,),
        dtype=complex,
    )
    for group_index, (_key, rows, cols, value_slice) in enumerate(
        coords.iter_key_coordinates()
    ):
        selected = np.einsum(
            "...pa,...a,...pa->...p",
            eigenvectors[..., rows, :],
            occupation,
            eigenvectors[..., cols, :].conj(),
            optimize=True,
        )
        if phases is not None:
            selected = selected * phases[..., group_index, np.newaxis]
        values[..., value_slice] = selected
    return values
