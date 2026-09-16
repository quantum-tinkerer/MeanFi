"""Periodic grids and streamed evaluations, with bounded spectrum storage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from meanfi.density.integrate.methods import UniformGrid
from meanfi.density.kpoint.matrix_functions import DirectDiagonalization, RationalFOE
from meanfi.density.kpoint.matrix_functions.direct import (
    density_values_from_eigensystem,
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
    charge: float | None
    band_energy: float | None
    entropy: float | None

    def errors(self, other: _Integral):
        return (
            np.abs(self.values - other.values),
            None if self.charge is None else abs(self.charge - other.charge),
            None
            if self.band_energy is None
            else abs(self.band_energy - other.band_energy),
            None if self.entropy is None else abs(self.entropy - other.entropy),
        )


@dataclass
class _Work:
    diagonalizations: int = 0
    kernels: int = 0
    charge_calls: int = 0
    density_calls: int = 0
    spectrum_bytes: int = 0


class _Grid:
    """One tensor grid, with an optional retained normal-state spectrum."""

    def __init__(self, n: int, dimension: int):
        self.n = n
        self.dimension = dimension
        self.shape = (n,) * dimension
        self.count = n**dimension
        self.spectra: np.ndarray | None = None

    def batches(self, batch_size: int):
        for start in range(0, self.count, batch_size):
            flat = np.arange(start, min(start + batch_size, self.count))
            indices = (
                np.stack(np.unravel_index(flat, self.shape), axis=-1)
                if self.dimension
                else np.empty((len(flat), 0), dtype=int)
            )
            points = 2 * np.pi * indices / self.n
            yield flat, indices, points


def _electron_charge(vectors, occupation):
    """Trace only the physical electron block of a batched BdG density."""
    electron_vectors = vectors[:, : vectors.shape[1] // 2]
    return np.einsum(
        "bia,ba,bia->b",
        electron_vectors,
        occupation,
        electron_vectors.conj(),
        optimize=True,
    ).real


class _Evaluator:
    def __init__(
        self,
        hamiltonian: _tb_type,
        *,
        kT: float,
        integration: UniformGrid,
        coordinates: DensityCoordinates,
        q_diag: np.ndarray | None,
        tolerances: ErrorTolerances,
        sparse_layout: SparseRationalLayout | None,
        fixed_filling: bool = False,
        compute_entropy: bool = False,
    ):
        self.compute_entropy = compute_entropy
        self.hamiltonian = hamiltonian
        self.kT = kT
        self.integration = integration
        self.coordinates = coordinates
        self.normal = q_diag is None
        self.size = tb_orbital_count(hamiltonian)
        self.q_diag = np.ones(self.size) if q_diag is None else np.asarray(q_diag)
        self.dtype = integration.dtype
        self.batch_size = integration.batch_size
        self.tolerances = tolerances
        self.fixed_filling = fixed_filling
        self.method = integration.matrix_function
        self.sparse_layout = sparse_layout
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
        self.entropy_approximation_error = None
        self.matrix_function_error = None

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
                "UniformGrid spectrum-storage limit reached: refinement requires "
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

    def _rational_node(self, matrix, *, compute_entropy=False):
        return PreparedMumpsRationalNode(
            matrix,
            kT=self.kT,
            q_diag=self.q_diag,
            options=self.method,
            layout=self.sparse_layout,
            matrix_function_tol=self.tolerances.matrix_function_tol,
            workspace_dtype=self.dtype,
            shared_aaa_interval_cache=self._aaa_interval_cache,
            compute_entropy=compute_entropy,
        )

    def charge(self, grid: _Grid, mu: float) -> tuple[float, float | None]:
        self.work.charge_calls += 1
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
                charges = (
                    occupation.sum(axis=1)
                    if self.normal
                    else _electron_charge(vectors, occupation)
                )
                total += float(charges.sum())
                self.work.diagonalizations += len(points)
                self.work.kernels += len(points)
        return total / grid.count, None

    def density(self, grid: _Grid, mu: float, *, compare_previous: bool):
        """Integrate requested entries and reuse available data for observables."""
        values = np.zeros(self.coordinates.value_count, dtype=complex)
        previous_values = np.zeros_like(values)
        charge = previous_charge = energy = previous_energy = 0.0
        entropy = previous_entropy = 0.0
        direct = isinstance(self.method, DirectDiagonalization)
        compute_energy = (
            self.fixed_filling or self.compute_entropy or (direct and self.normal)
        )
        compute_charge = self.fixed_filling or (
            direct and (self.normal or compute_energy)
        )
        self.work.density_calls += 1
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
                if not self.normal and compute_energy
                else None
            )
            if not direct:
                packed = np.empty(
                    (len(points), self.coordinates.value_count), dtype=complex
                )
                charges = np.empty(len(points)) if compute_charge else None
                energies = np.empty(len(points)) if compute_energy else None
                entropies = np.empty(len(points)) if self.compute_entropy else None
                for index, matrix in enumerate(matrices):
                    node = self._rational_node(
                        matrix, compute_entropy=self.compute_entropy
                    )
                    packed[index] = node.density_values(mu)
                    if compute_energy:
                        energies[index], point_entropy = node.thermodynamics(mu)
                        if self.compute_entropy:
                            entropies[index] = point_entropy
                    if compute_charge:
                        charges[index] = node.charge(mu)
                    if node.matrix_function_error is not None:
                        self.matrix_function_error = max(
                            self.matrix_function_error or 0.0,
                            node.matrix_function_error,
                        )
                    if self.compute_entropy:
                        self.entropy_approximation_error = max(
                            self.entropy_approximation_error or 0.0,
                            node._last_terms.entropy_error,
                        )
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
                packed = density_values_from_eigensystem(
                    vectors, occupation, self.coordinates, phases=phases
                )
                charges = energies = None
                if self.normal:
                    charges = occupation.sum(axis=1)
                elif compute_charge:
                    charges = _electron_charge(vectors, occupation)
                if compute_energy:
                    energies = np.sum(eigenvalues * occupation, axis=1)
                    if not self.normal:
                        energies += mu * (2 * charges - occupation.sum(axis=1))
                entropies = (
                    occupation_entropy(occupation).sum(axis=1)
                    if self.compute_entropy
                    else None
                )
                self.work.diagonalizations += len(points)
                self.work.kernels += len(points)
            if compute_energy:
                if not self.normal:
                    energies += electron_trace
                # Nambu size includes the half factor and physical-orbital normalization.
                energies /= self.size
                energy += float(energies.sum())
            if self.compute_entropy:
                entropies /= self.size
                entropy += float(entropies.sum())
            values += packed.sum(axis=0)
            if compute_charge:
                charge += float(charges.sum())
            if compare_previous:
                old = np.all(indices % 2 == 0, axis=1)
                previous_values += packed[old].sum(axis=0)
                if compute_charge:
                    previous_charge += float(charges[old].sum())
                if compute_energy:
                    previous_energy += float(energies[old].sum())
                if self.compute_entropy:
                    previous_entropy += float(entropies[old].sum())
        previous_count = (grid.n // 2) ** grid.dimension if compare_previous else 1
        return (
            _Integral(
                values / grid.count,
                charge / grid.count if compute_charge else None,
                energy / grid.count if compute_energy else None,
                entropy / grid.count if self.compute_entropy else None,
            ),
            _Integral(
                previous_values / previous_count,
                previous_charge / previous_count if compute_charge else None,
                previous_energy / previous_count if compute_energy else None,
                previous_entropy / previous_count if self.compute_entropy else None,
            ),
        )
