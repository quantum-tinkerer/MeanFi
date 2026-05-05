from __future__ import annotations

from typing import Any

import numpy as np

from meanfi.tb.ops import to_dense
from meanfi.integrate.occupations import fermi_dirac

from .base import _BlockResult


def _exact_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    workspace_dtype: np.dtype,
) -> _BlockResult:
    array = np.asarray(to_dense(matrix), dtype=workspace_dtype)
    block = np.asarray(block, dtype=workspace_dtype)
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    occupation = fermi_dirac(eigenvalues, kT, 0.0)
    projected_block = eigenvectors.conj().T @ block
    density_result = eigenvectors @ (occupation[:, np.newaxis] * projected_block)

    derivative_block = None
    if derivative:
        fprime = -occupation * (1.0 - occupation) / kT
        delta = eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]
        numerator = occupation[:, np.newaxis] - occupation[np.newaxis, :]
        loewner = np.empty_like(delta, dtype=float)
        separated = np.abs(delta) > 1e-12
        loewner[separated] = numerator[separated] / delta[separated]
        loewner[~separated] = np.broadcast_to(fprime[:, np.newaxis], delta.shape)[
            ~separated
        ]

        d_h = -(q_diag[:, np.newaxis] * eigenvectors)
        projected_dh = eigenvectors.conj().T @ d_h
        derivative_block = eigenvectors @ ((loewner * projected_dh) @ projected_block)

    return _BlockResult(
        block=density_result,
        derivative_block=derivative_block,
        error=0.0,
        order=None,
    )
