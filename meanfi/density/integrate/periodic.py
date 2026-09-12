"""Periodic sampling with streamed density evaluation and global refinement.

Only normal-state eigenvalues survive an evaluation. Hamiltonians and
vectors occupy at most one batch; BdG spectra are recomputed whenever mu changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from meanfi.density.filling import mu_bracket, mu_bracket_for_bdg, solve_mu
from meanfi.density.integrate.methods import PeriodicGrid
from meanfi.density.integrate.workspace import workspace_complex_dtype
from meanfi.density.internal import DensityEvaluation, DensitySlice
from meanfi.density.kpoint.matrix_functions import (
    DirectDiagonalization,
    RationalFOE,
    resolve_matrix_function,
    selected_density_values_from_eigensystem,
)
from meanfi.density.kpoint.matrix_functions.rational import PreparedMumpsRationalNode
from meanfi.density.kpoint.occupations import fermi_dirac
from meanfi.errors import ErrorTolerances, ErrorValues, default_solver_tolerances
from meanfi.results import PeriodicGridInfo
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates
from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like, to_dense
from meanfi.tb.validate import tb_dimension, tb_orbital_count


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


def resolve_periodic_matrix_function(
    selected: object | None,
    hamiltonian: _tb_type,
    *,
    kT: float,
    prescribed: bool,
) -> DirectDiagonalization | RationalFOE:
    sparse = any(is_sparse_like(matrix) for matrix in hamiltonian.values())
    if selected is None:
        if sparse:
            if prescribed and kT > 0:
                return RationalFOE(rational_scheme="aaa")
            raise ValueError(
                "Automatic sparse PeriodicGrid evaluation requires kT > 0 and "
                "prescribed nk. Use PeriodicGrid(nk=..., matrix_function=RationalFOE()) "
                "or explicitly choose DirectDiagonalization() to permit dense batches."
            )
        return DirectDiagonalization()
    resolved = resolve_matrix_function(selected)
    if isinstance(resolved, RationalFOE):
        if not prescribed:
            raise ValueError(
                "PeriodicGrid RationalFOE requires prescribed nk; adaptive RationalFOE is unsupported"
            )
        if kT <= 0:
            raise ValueError("PeriodicGrid RationalFOE requires kT > 0")
        if not sparse:
            raise ValueError(
                "PeriodicGrid RationalFOE is supported only for sparse matrices"
            )
    elif not isinstance(resolved, DirectDiagonalization):
        raise TypeError(
            "PeriodicGrid.matrix_function must be DirectDiagonalization or RationalFOE"
        )
    return resolved


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
        self.dtype = workspace_complex_dtype(integration)
        self.batch_size = integration.batch_size or 128
        self.tolerances = tolerances
        self.method = resolve_periodic_matrix_function(
            integration.matrix_function,
            hamiltonian,
            kT=kT,
            prescribed=integration.nk is not None,
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
            density_coordinates=self.coordinates,
            density_tolerance=self.tolerances.density_matrix_integration,
            workspace_dtype=self.dtype,
            trace_weights_diag=self.trace_weights,
            shared_aaa_interval_cache=self._aaa_interval_cache,
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
                    total += node.charge_and_derivative(mu)[0]
                    self.work.kernels += 1
                    # Retain one scalar fit, never matrices or factorizations.
                    del self._aaa_interval_cache[:-1]
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
        self.work.density_calls += 1
        self.work.evaluations += grid.count
        for _flat, indices, points in grid.batches(self.batch_size):
            matrices = self.matrices_at(points)
            phases = np.exp(1j * (points @ self.density_keys.T))
            if isinstance(self.method, RationalFOE):
                packed = np.empty(
                    (len(points), self.coordinates.value_count), dtype=complex
                )
                charges = np.empty(len(points))
                for index, matrix in enumerate(matrices):
                    node = self._rational_node(matrix)
                    charges[index] = node.charge_and_derivative(mu)[0]
                    packed[index] = node.density_values_from_charge_order(mu)
                    for group, (_key, _rows, _cols, value_slice) in enumerate(
                        self.coordinates.iter_key_coordinates()
                    ):
                        packed[index, value_slice] *= phases[index, group]
                    self.work.kernels += 1
                    # Retain one scalar fit, never matrices or factorizations.
                    del self._aaa_interval_cache[:-1]
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
                self.work.diagonalizations += len(points)
                self.work.kernels += len(points)
            values += packed.sum(axis=0)
            charge += float(charges.sum())
            if compare_previous:
                old = np.all(indices % 2 == 0, axis=1)
                previous_values += packed[old].sum(axis=0)
                previous_charge += float(charges[old].sum())
        if compare_previous:
            previous_count = (grid.n // 2) ** grid.dimension
            previous_values /= previous_count
            previous_charge /= previous_count
        return (
            values / grid.count,
            charge / grid.count,
            previous_values,
            previous_charge,
        )


def solve_periodic(
    hamiltonian: _tb_type,
    *,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: PeriodicGrid,
    mu: float | None = None,
    filling: float | None = None,
    filling_tol: float | None = None,
    mu_tol: float = 1e-12,
    max_charge_evaluations: int | None = None,
    mu_guess: float = 0.0,
    density_coordinates: DensityCoordinates | None = None,
    q_diag: np.ndarray | None = None,
    trace_weights_diag: np.ndarray | None = None,
    tolerances: ErrorTolerances | None = None,
) -> DensityEvaluation:
    """Evaluate one prescribed grid, or refine until nested and shifted tests pass."""
    if not np.isfinite(kT) or kT < 0:
        raise ValueError("kT must be finite and non-negative")
    if (mu is None) == (filling is None):
        raise ValueError("Provide exactly one of mu and filling")
    if max_charge_evaluations is not None and (
        isinstance(max_charge_evaluations, bool)
        or not isinstance(max_charge_evaluations, Integral)
        or max_charge_evaluations <= 0
    ):
        raise ValueError(
            "max_charge_evaluations must be a positive integer when provided"
        )
    prescribed = integration.nk is not None
    if not prescribed and kT <= 0:
        raise ValueError(
            "Accuracy-controlled PeriodicGrid requires kT > 0; use PeriodicGrid(nk=...) at T=0"
        )
    if mu is not None and not np.isfinite(mu):
        raise ValueError("mu must be finite")
    dimension = tb_dimension(hamiltonian)
    size = tb_orbital_count(hamiltonian)
    coordinates = density_coordinates or full_density_coordinates(keys, size=size)
    tolerances = tolerances or default_solver_tolerances(1e-5)
    density_target = (
        integration.density_matrix_tol or tolerances.density_matrix_integration
    )
    charge_target = integration.charge_tol or tolerances.charge_integration
    filling_tol = tolerances.filling_residual if filling_tol is None else filling_tol
    # Pointwise rational accuracy must respect an explicit root tolerance too.
    tolerances = replace(tolerances, filling_residual=filling_tol)
    weights = (
        np.ones(size)
        if trace_weights_diag is None
        else np.asarray(trace_weights_diag, dtype=float)
    )
    evaluator = _Evaluator(
        hamiltonian,
        kT=kT,
        integration=integration,
        coordinates=coordinates,
        q_diag=q_diag,
        trace_weights=weights,
        tolerances=tolerances,
    )
    n = (
        periodic_grid_resolution(integration.nk, dimension)
        if prescribed
        else (4 if dimension else 1)
    )
    previous = None
    refinements = 0
    charge_evaluations = 0
    density_error = charge_error = None
    while True:
        count = n**dimension
        if count > integration.max_points:
            raise RuntimeError(
                "PeriodicGrid total-grid-size limit reached: "
                f"the next mesh needs {count} points, max_points={integration.max_points}. "
                "Increase max_points or reduce nk / relax integration targets."
            )
        grid = _Grid(n, dimension)
        # Primary grids nest; the irrational validation shifts remain distinct.
        evaluator.work.unique = count + evaluator.work.validation_points
        if filling is not None:
            evaluator.retain_spectra(grid, previous)
            bracket = mu_bracket if q_diag is None else mu_bracket_for_bdg
            remaining = (
                None
                if max_charge_evaluations is None
                else max_charge_evaluations - charge_evaluations
            )
            if remaining is not None and remaining <= 0:
                raise RuntimeError(
                    "PeriodicGrid chemical-potential solve reached max_charge_evaluations across refinement grids"
                )
            root = solve_mu(
                evaluate_charge=lambda candidate: evaluator.charge(grid, candidate),
                initial_bracket=lambda: bracket(hamiltonian, kT),
                filling=filling,
                mu_guess=mu_guess,
                filling_tol=filling_tol,
                mu_tol=mu_tol,
                max_charge_evaluations=remaining,
                use_derivative=evaluator.normal
                and isinstance(evaluator.method, DirectDiagonalization)
                and kT > 0,
            )
            charge_evaluations += root.charge_evaluations
            resolved_mu = mu_guess = root.mu
        else:
            resolved_mu = float(mu)
        values, charge, old_values, old_charge = evaluator.density(
            grid, resolved_mu, compare_previous=previous is not None
        )
        if prescribed:
            break
        if dimension == 0:
            density_error = np.zeros(values.size)
            charge_error = 0.0
            break
        if previous is not None:
            density_error = np.abs(values - old_values)
            charge_error = abs(charge - old_charge)
            if (
                np.max(density_error, initial=0.0) <= density_target
                and charge_error <= charge_target
            ):
                shifted = _Grid(n, dimension, shifted=True)
                shifted_values, shifted_charge, _, _ = evaluator.density(
                    shifted, resolved_mu, compare_previous=False
                )
                evaluator.work.unique += count
                evaluator.work.validation_points += count
                density_error = np.maximum(
                    density_error, np.abs(values - shifted_values)
                )
                charge_error = max(charge_error, abs(charge - shifted_charge))
                if (
                    np.max(density_error, initial=0.0) <= density_target
                    and charge_error <= charge_target
                ):
                    break
        if (
            integration.max_refinements is not None
            and refinements >= integration.max_refinements
        ):
            raise RuntimeError(
                "PeriodicGrid did not converge before max_refinements="
                f"{integration.max_refinements}; mesh={grid.shape}, "
                f"density_error={None if density_error is None else np.max(density_error, initial=0.0)}, "
                f"charge_error={charge_error}. Increase the limit or relax integration targets."
            )
        previous = grid
        n *= 2
        refinements += 1
    if filling is not None and abs(charge - filling) > filling_tol:
        raise RuntimeError(
            "PeriodicGrid density recomputation did not satisfy the filling tolerance: "
            f"residual={abs(charge - filling)}, filling_tol={filling_tol}. "
            "Use workspace_precision=128 or tighten the matrix-function accuracy."
        )
    work = evaluator.work
    info = PeriodicGridInfo(
        requested_nk=integration.nk,
        n_kpoints=grid.count,
        grid_shape=grid.shape,
        n_kernel_evals=work.kernels,
        n_diagonalizations=(
            work.diagonalizations
            if isinstance(evaluator.method, DirectDiagonalization)
            else None
        ),
        unique_evals=work.unique,
        n_evaluator_evals=work.evaluations,
        refinements=refinements,
        validation_evaluations=work.validation_points,
        charge_evaluations=charge_evaluations,
        charge_integration_calls=work.charge_calls,
        density_integration_calls=work.density_calls,
        charge_error=charge_error,
        error_estimate_available=not prescribed,
        spectrum_bytes=work.spectrum_bytes,
    )
    return DensityEvaluation(
        density=DensitySlice(coordinates, values, density_error),
        mu=resolved_mu,
        filling=charge,
        errors=ErrorValues(
            density_matrix_integration=None
            if density_error is None
            else float(np.max(density_error, initial=0.0)),
            charge_integration=charge_error,
            filling_residual=None if filling is None else abs(charge - filling),
        ),
        integration=integration,
        statistics=info,
    )
