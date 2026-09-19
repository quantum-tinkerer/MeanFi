"""Degree-two spectral energy quadrature on the exported density mesh."""

from dataclasses import dataclass
from math import gcd

import numpy as np


@dataclass(frozen=True)
class SimplexEnergy:
    band_energy: float
    evaluations: int = 0
    simplices: int = 0


def integrate_energies(mesh, *, mu, filling, batch_size=256):
    """Integrate the continuous spectral hinge, then restore mu times charge.

    In dimension d the normalized weights are 1/((d+1)(d+2)) at each
    vertex and (d+1)/(d+2) at the centroid. The finest complete evaluated
    partition and all retained spectra, including previews, are reused;
    only unique, uncached centroids require eigenvalues. Bounded NumPy batches
    avoid storing all centroid Hamiltonians. No eigenvectors or refinement.

    ``filling`` is the trace on the density integration partition, not the
    requested filling. The result is per orbital, like all MeanFi energies.
    """
    snapshot = mesh.evaluated_snapshot(include_eigenvectors=False)
    simplices = snapshot.simplices
    eigenvalues = snapshot.eigenvalues
    size = snapshot.ndof
    dimension = snapshot.ndim
    vertex_values = np.minimum(eigenvalues - mu, 0).mean(axis=1)
    centers, center_indices = _centroid_samples(snapshot)
    values = np.empty(len(vertex_values) + len(centers))
    values[: len(vertex_values)] = vertex_values
    for start in range(0, len(centers), batch_size):
        batch = centers[start : start + batch_size]
        matrices = np.asarray([mesh.evaluate(*point) for point in batch])
        spectra = np.linalg.eigvalsh(matrices)
        offset = len(vertex_values) + start
        values[offset : offset + len(batch)] = np.minimum(spectra - mu, 0).mean(axis=1)
    integral = snapshot.volumes @ (
        vertex_values[simplices].mean(axis=1) / (dimension + 2)
        + (dimension + 1) / (dimension + 2) * values[center_indices]
    )
    if filling is None:
        # Empty interaction support has no stored density to trace. Reuse the
        # occupation integral on the exported mesh for this noninteracting case.
        filling = float(np.sum(mesh.occupied_weights(float(mu))))
    return SimplexEnergy(
        float(integral + mu * filling / size), len(centers), len(simplices)
    )


def _rational_key(numerators, denominator):
    """Canonical exact coordinates with one positive common denominator."""
    divisor = gcd(denominator, *numerators)
    return tuple(value // divisor for value in numerators), denominator // divisor


def _centroid_samples(snapshot):
    """Index centroids into the complete cache followed by unique new points.

    Integer dyadic coordinates avoid rounding-dependent duplicate tests. Cache
    rows outside the exported partition remain eligible for reuse.
    """
    numerators = [tuple(map(int, row)) for row in snapshot.dyadic_numerators]
    levels = list(map(int, snapshot.dyadic_levels))
    cache = {
        _rational_key(row, 1 << level): index
        for index, (row, level) in enumerate(zip(numerators, levels, strict=True))
    }
    centers = []
    indices = []
    for simplex in snapshot.simplices:
        level = max(levels[index] for index in simplex)
        numerator = tuple(
            sum(numerators[index][axis] << (level - levels[index]) for index in simplex)
            for axis in range(snapshot.ndim)
        )
        key = _rational_key(numerator, len(simplex) << level)
        if key not in cache:
            cache[key] = len(numerators) + len(centers)
            centers.append([value / key[1] for value in key[0]])
        indices.append(cache[key])
    return np.asarray(centers).reshape(-1, snapshot.ndim), np.asarray(indices)
