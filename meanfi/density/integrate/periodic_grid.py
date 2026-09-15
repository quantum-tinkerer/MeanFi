"""Periodic grids and streamed evaluations, with bounded spectrum storage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from meanfi.density.integrate.methods import PeriodicGrid
from meanfi.density.kpoint.matrix_functions import DirectDiagonalization, RationalFOE
from meanfi.density.kpoint.matrix_functions.direct import (
    selected_density_values_from_eigensystem,
)
from meanfi.density.kpoint.matrix_functions.rational import PreparedMumpsRationalNode
from meanfi.density.kpoint.matrix_functions.rational.common import SparseRationalLayout
from meanfi.density.kpoint.occupations import fermi_dirac, occupation_entropy
from meanfi.errors import ErrorTolerances
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import _tb_type, as_sparse, to_dense
from meanfi.tb.validate import tb_orbital_count


def periodic_grid_resolution(nk: int, dimension: int) -> int:
    """Smallest isotropic axis length whose total point count reaches nk."""
    if dimension == 0:
        return 1
    n = max(1, int(round(nk ** (1.0 / dimension))))
    while n**dimension < nk:
        n += 1
    while n > 1 and (n - 1) ** dimension >= nk:
        n -= 1
    return n


@dataclass
class _Integral:
    values: np.ndarray
    charge: float
    band_energy: float
    entropy: float

    def errors(self, other: _Integral):
        return (
            np.abs(self.values - other.values),
            abs(self.charge - other.charge),
            abs(self.band_energy - other.band_energy),
            abs(self.entropy - other.entropy),
        )


@dataclass
class _Work:
    diagonalizations: int = 0
    kernels: int = 0
    evaluations: int = 0
    unique: int = 0
    charge_calls: int = 0
    density_calls: int = 0
    validation_points: int = 0
    spectrum_bytes: int = 0


def _validation_shift(dimension: int) -> np.ndarray:
    # Distinct irrational offsets avoid the simple aliases shared by dyadic
    # grids and half-cell shifts. They do not make the estimate rigorous.
    primes = []
    candidate = 2
    while len(primes) < dimension:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return np.sqrt(primes) % 1.0


class _Grid:
    """One tensor grid, with an optional retained normal-state spectrum."""

    def __init__(self, n: int, dimension: int, *, shifted: bool = False):
        self.n = n
        self.dimension = dimension
        self.shape = (n,) * dimension
        self.count = n**dimension
        self.shift = _validation_shift(dimension) if shifted else 0.0
        self.spectra: np.ndarray | None = None

    def batches(self, batch_size: int):
        for start in range(0, self.count, batch_size):
            flat = np.arange(start, min(start + batch_size, self.count))
            indices = (
                np.stack(np.unravel_index(flat, self.shape), axis=-1)
                if self.dimension
                else np.empty((len(flat), 0), dtype=int)
            )
            points = 2 * np.pi * (indices + self.shift) / self.n
            yield flat, indices, points


class _Evaluator:
    def __init__(
        self,
        hamiltonian: _tb_type,
        *,
        kT: float,
        integration: PeriodicGrid,
        coordinates: DensityCoordinates,
        q_diag: np.ndarray | None,
        trace_weights: np.ndarray,
        tolerances: ErrorTolerances,
    ):
        self.hamiltonian = hamiltonian
        self.kT = kT
        self.integration = integration
        self.coordinates = coordinates
        self.normal = q_diag is None
        self.size = tb_orbital_count(hamiltonian)
        self.q_diag = np.ones(self.size) if q_diag is None else np.asarray(q_diag)
        self.trace_weights = trace_weights
        self.dtype = integration.dtype
        self.batch_size = integration.batch_size or 128
        self.tolerances = tolerances
        self.method = integration.matrix_function
        self.sparse_layout = (
            SparseRationalLayout.build(
                density_coordinates=coordinates,
                trace_weights_diag=trace_weights,
                include_all_diagonal=True,
            )
            if isinstance(self.method, RationalFOE)
            else None
        )
        self.tb_keys = np.asarray(list(hamiltonian), dtype=float)
        self.density_keys = np.asarray(coordinates.keys, dtype=float)
        self.matrices = (
            [
                as_sparse(matrix).astype(self.dtype).tocsr()
                for matrix in hamiltonian.values()
            ]
            if isinstance(self.method, RationalFOE)
            else np.asarray(
                [to_dense(matrix) for matrix in hamiltonian.values()], dtype=self.dtype
            )
        )
        self.work = _Work()
        self._aaa_interval_cache = []

    def matrices_at(self, points: np.ndarray):
        phases = np.exp(-1j * (points @ self.tb_keys.T))
        if isinstance(self.method, DirectDiagonalization):
            return np.einsum(
                "bk,kij->bij", phases, self.matrices, optimize=True
            ).astype(self.dtype, copy=False)
        return [
            sum(
                (
                    phase * matrix
                    for phase, matrix in zip(row, self.matrices, strict=True)
                )
            )
            for row in phases
        ]

    def retain_spectra(self, grid: _Grid, previous: _Grid | None) -> None:
        if not self.normal or not isinstance(self.method, DirectDiagonalization):
            return
        real_dtype = np.empty((), dtype=self.dtype).real.dtype
        required = grid.count * self.size * real_dtype.itemsize
        previous_bytes = (
            0
            if previous is None or previous.spectra is None
            else previous.spectra.nbytes
        )
        if required + previous_bytes > self.integration.max_spectrum_bytes:
            raise RuntimeError(
                "PeriodicGrid spectrum-storage limit reached: refinement requires "
                f"{required + previous_bytes} retained bytes, max_spectrum_bytes="
                f"{self.integration.max_spectrum_bytes}. Increase max_spectrum_bytes "
                "or reduce the prescribed nk / relax integration targets."
            )
        spectra = np.empty((grid.count, self.size), dtype=real_dtype)
        self.work.spectrum_bytes = max(
            self.work.spectrum_bytes, required + previous_bytes
        )
        for flat, indices, points in grid.batches(self.batch_size):
            old = np.zeros(flat.size, dtype=bool)
            if previous is not None and previous.spectra is not None:
                old = np.all(indices % 2 == 0, axis=1)
                coarse = np.ravel_multi_index((indices[old] // 2).T, previous.shape)
                spectra[flat[old]] = previous.spectra[coarse]
            new = ~old
            if np.any(new):
                spectra[flat[new]] = np.linalg.eigvalsh(self.matrices_at(points[new]))
                count = int(np.count_nonzero(new))
                self.work.diagonalizations += count
                self.work.kernels += count
        grid.spectra = spectra
        if previous is not None:
            previous.spectra = None

    def _rational_node(self, matrix):
        return PreparedMumpsRationalNode(
            matrix,
            kT=self.kT,
            q_diag=self.q_diag,
            options=self.method,
            charge_tolerance=min(
                self.tolerances.charge_integration, self.tolerances.filling_residual / 4
            ),
            layout=self.sparse_layout,
            density_tolerance=self.tolerances.density_matrix_integration,
            workspace_dtype=self.dtype,
            shared_aaa_interval_cache=self._aaa_interval_cache,
            band_energy_tolerance=self.size * self.tolerances.band_energy_integration,
            entropy_tolerance=self.size * self.tolerances.entropy_integration,
        )

    def charge(self, grid: _Grid, mu: float) -> tuple[float, float, float | None]:
        self.work.charge_calls += 1
        self.work.evaluations += grid.count
        if grid.spectra is not None:
            total = derivative = 0.0
            for start in range(0, grid.count, self.batch_size):
                occupation = fermi_dirac(
                    grid.spectra[start : start + self.batch_size], self.kT, mu
                )
                total += float(np.sum(occupation))
                if self.kT > 0:
                    derivative += float(np.sum(occupation * (1 - occupation)) / self.kT)
            return (
                total / grid.count,
                0.0,
                derivative / grid.count if self.kT > 0 else None,
            )
        total = 0.0
        for _flat, _indices, points in grid.batches(self.batch_size):
            matrices = self.matrices_at(points)
            if isinstance(self.method, RationalFOE):
                for matrix in matrices:
                    node = self._rational_node(matrix)
                    total += node.charge(mu)
                    self.work.kernels += 1
                    del node
            else:
                matrices[:, np.arange(self.size), np.arange(self.size)] -= (
                    mu * self.q_diag
                )
                values, vectors = np.linalg.eigh(matrices)
                occupation = fermi_dirac(values, self.kT, 0.0)
                diagonal = np.einsum(
                    "bia,ba,bia->bi", vectors, occupation, vectors.conj(), optimize=True
                ).real
                total += float(np.sum(diagonal @ self.trace_weights))
                self.work.diagonalizations += len(points)
                self.work.kernels += len(points)
        return total / grid.count, 0.0, None

    def density(self, grid: _Grid, mu: float, *, compare_previous: bool):
        """Evaluate requested entries and, optionally, the nested parent at this mu."""
        values = np.zeros(self.coordinates.value_count, dtype=complex)
        previous_values = np.zeros_like(values)
        charge = previous_charge = 0.0
        energy = previous_energy = entropy = previous_entropy = 0.0
        self.work.density_calls += 1
        self.work.evaluations += grid.count
        for _flat, indices, points in grid.batches(self.batch_size):
            matrices = self.matrices_at(points)
            phases = np.exp(1j * (points @ self.density_keys.T))
            electron_trace = (
                np.asarray(
                    [
                        matrix.diagonal()[: self.size // 2].sum().real
                        for matrix in matrices
                    ]
                )
                if not self.normal
                else None
            )
            if isinstance(self.method, RationalFOE):
                packed = np.empty(
                    (len(points), self.coordinates.value_count), dtype=complex
                )
                charges = np.empty(len(points))
                energies = np.empty(len(points))
                entropies = np.empty(len(points))
                for index, matrix in enumerate(matrices):
                    node = self._rational_node(matrix)
                    charges[index] = node.charge(mu)
                    packed[index] = node.density_values_from_charge_order(mu)
                    energies[index], entropies[index] = node.thermodynamics(mu)
                    for group, (_key, _rows, _cols, value_slice) in enumerate(
                        self.coordinates.iter_key_coordinates()
                    ):
                        packed[index, value_slice] *= phases[index, group]
                    self.work.kernels += 1
                    del node
            else:
                if not self.normal:
                    matrices[:, np.arange(self.size), np.arange(self.size)] -= (
                        mu * self.q_diag
                    )
                eigenvalues, vectors = np.linalg.eigh(matrices)
                occupation = fermi_dirac(
                    eigenvalues, self.kT, mu if self.normal else 0.0
                )
                if any(rows.size > self.size for rows in self.coordinates.rows_by_key):
                    # Full layouts must not create a batch x size**3 temporary.
                    density = (
                        vectors * occupation[:, None, :]
                    ) @ vectors.conj().swapaxes(-1, -2)
                    packed = np.empty(
                        (len(points), self.coordinates.value_count), dtype=complex
                    )
                    for group, (_key, rows, cols, value_slice) in enumerate(
                        self.coordinates.iter_key_coordinates()
                    ):
                        packed[:, value_slice] = (
                            density[:, rows, cols] * phases[:, group, None]
                        )
                else:
                    packed = selected_density_values_from_eigensystem(
                        vectors, occupation, self.coordinates, phases=phases
                    )
                diagonal = np.einsum(
                    "bia,ba,bia->bi", vectors, occupation, vectors.conj(), optimize=True
                ).real
                charges = diagonal @ self.trace_weights
                energies = np.sum(eigenvalues * occupation, axis=1)
                entropies = occupation_entropy(occupation).sum(axis=1)
                if not self.normal:
                    energies += mu * (diagonal @ self.q_diag)
                self.work.diagonalizations += len(points)
                self.work.kernels += len(points)
            if not self.normal:
                energies += electron_trace
            # Dividing by the Nambu size includes both its half factor and
            # normalization by the number of physical orbitals.
            energies /= self.size
            entropies /= self.size
            values += packed.sum(axis=0)
            charge += float(charges.sum())
            energy += float(energies.sum())
            entropy += float(entropies.sum())
            if compare_previous:
                old = np.all(indices % 2 == 0, axis=1)
                previous_values += packed[old].sum(axis=0)
                previous_charge += float(charges[old].sum())
                previous_energy += float(energies[old].sum())
                previous_entropy += float(entropies[old].sum())
        previous_count = (grid.n // 2) ** grid.dimension if compare_previous else 1
        return (
            _Integral(
                values / grid.count,
                charge / grid.count,
                energy / grid.count,
                entropy / grid.count,
            ),
            _Integral(
                previous_values / previous_count,
                previous_charge / previous_count,
                previous_energy / previous_count,
                previous_entropy / previous_count,
            ),
        )
