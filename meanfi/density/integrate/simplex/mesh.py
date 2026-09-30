"""FermiSimplex operations, resource bounds, and cumulative work."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import comb, factorial
from functools import wraps

import numpy as np
from fermisimplex import SpectralMesh
from threadpoolctl import threadpool_limits

from meanfi.hamiltonian import BlochHamiltonian, Hamiltonian, hamiltonian_dimension
from meanfi.density.problem import DensityProblem
from meanfi.results import IntegrationInfo, _DensityEntries
from meanfi.space.coordinates import DensityCoordinates
from meanfi.tb.ops import to_dense

_CHARGE_ERROR_DEPTH = 2
_MIN_REFINEMENT_BATCH_SIZE = 1
_MAX_REFINEMENT_BATCH_SIZE = 100


def _spectral_mesh(
    h: Hamiltonian,
    *,
    nk: int | None = None,
    max_points: int | None = None,
) -> SpectralMesh:
    # FermiSimplex's dyadic root mesh includes both faces of the unit cell.
    # Its native construction gives (2**level + 1)**dimension distinct nodes.
    dimension = hamiltonian_dimension(h)
    # Three vertices alias the first cosine harmonic with its midpoint preview.
    level = 2 if nk is None else 0
    while nk is not None and (2**level + 1) ** dimension < nk:
        level += 1
    nodes = (2**level + 1) ** dimension
    if max_points is not None and nodes > max_points:
        raise RuntimeError(
            f"FermiSimplex mesh requires {nodes} nodes for nk={nk}, "
            f"exceeding max_points={max_points}; increase max_points or reduce nk"
        )
    if isinstance(h, BlochHamiltonian):

        @wraps(h.function)
        def evaluate(*coordinates):
            return h(*(2 * np.pi * np.asarray(coordinates)))

        return SpectralMesh(evaluate, root_level=level)
    dense_hamiltonian = {
        key: np.asarray(to_dense(matrix), dtype=np.complex128)
        for key, matrix in h.items()
    }
    return SpectralMesh(dense_hamiltonian, root_level=level)


def _bounded_refinements(
    mesh: SpectralMesh,
    max_refinements: int | None,
    max_points: int | None,
    *,
    preview_depth: int,
) -> int | None:
    """Conservatively bound native retained spectra, including density previews.

    Each native refinement splits one simplex into 2**dimension children.
    Reserving their vertices before the call avoids an unbounded native cache
    without implementing another refinement engine. Shared vertices can make
    this reserve larger than the actual allocation.
    """
    if max_points is None:
        return max_refinements
    dimension = int(mesh.ndim)
    nodes_per_simplex = comb(dimension + 2**preview_depth, dimension)
    initial = max(int(mesh.cached_vertices), int(mesh.active_vertices))
    initial += int(mesh.active_simplices) * (nodes_per_simplex - dimension - 1)
    if initial > max_points:
        raise RuntimeError(
            f"FermiSimplex needs a reserve of {initial} cached/preview nodes, "
            f"exceeding max_points={max_points}; increase max_points or loosen tolerances"
        )
    per_refinement = 2**dimension * nodes_per_simplex
    available = (max_points - initial) // per_refinement
    return available if max_refinements is None else min(max_refinements, available)


def _bounded_density_bisections(
    mesh: SpectralMesh,
    max_refinements: int | None,
    max_points: int | None,
) -> int | None:
    """One density bisection adds at most one temporary midpoint spectrum."""
    if max_points is None:
        return max_refinements
    available = max_points - int(mesh.cached_vertices)
    if available < 0:
        raise RuntimeError(
            f"FermiSimplex already caches more than max_points={max_points} spectra"
        )
    return available if max_refinements is None else min(max_refinements, available)


def _native_thread_context(num_threads: int | None):
    if num_threads is None:
        return nullcontext()
    return threadpool_limits(limits=int(num_threads), user_api="openmp")


@contextmanager
def _integration_context(num_threads: int | None):
    try:
        with _native_thread_context(num_threads):
            yield
    except RuntimeError as error:
        if "did not converge" not in str(error):
            raise
        raise RuntimeError(
            f"{error}; increase max_points/max_refinements or loosen integration tolerances"
        ) from error


def _integrate_charge(
    mesh: SpectralMesh,
    *,
    mu: float,
    charge_tol: float,
    max_refinements: int | None,
    num_threads: int | None,
    max_points: int | None = None,
):
    max_refinements = _bounded_refinements(
        mesh, max_refinements, max_points, preview_depth=0
    )
    with _integration_context(num_threads):
        return mesh.integrate_charge(
            mu=float(mu),
            target_error=float(charge_tol),
            max_refinements=max_refinements,
            error_depth=_CHARGE_ERROR_DEPTH,
            min_refinement_batch_size=_MIN_REFINEMENT_BATCH_SIZE,
            max_refinement_batch_size=_MAX_REFINEMENT_BATCH_SIZE,
        )


def _evaluate_charge(
    mesh: SpectralMesh,
    *,
    mu: float,
    num_threads: int | None,
):
    cached_vertices = int(mesh.cached_vertices)
    with _native_thread_context(num_threads):
        result = mesh.estimate_charge_on_current_mesh(mu=float(mu))
    return result, int(mesh.cached_vertices) - cached_vertices


def _zero_temperature_entropy(
    mesh: SpectralMesh, mu: float, eigenvalues: np.ndarray | None = None
) -> float:
    """Only flat bands at mu have nonzero entropy in simplex integration."""
    if eigenvalues is None:
        eigenvalues = np.asarray(mesh.eigenvalues)
    at_mu = np.abs(eigenvalues - mu) <= mesh.tolerance
    if not np.any(at_mu):
        return 0.0
    simplices = np.asarray(mesh.simplices)
    counts = np.all(at_mu[simplices], axis=1).sum(axis=1)
    vertices = np.asarray(mesh.points)[simplices[counts > 0]]
    edges = vertices[:, 1:] - vertices[:, :1]
    volumes = np.abs(np.linalg.det(edges)) / factorial(mesh.ndim)
    return float(np.log(2.0) * (volumes @ counts[counts > 0])) / at_mu.shape[-1]


def _integrate_density(
    mesh: SpectralMesh,
    density_coordinates: DensityCoordinates,
    *,
    mu: float,
    density_atol: float,
    max_refinements: int | None,
    num_threads: int | None,
    prescribed: bool = False,
    max_points: int | None = None,
    max_degree: int = 7,
    max_h_refinements: int | None = None,
):
    if prescribed:
        max_refinements = _bounded_refinements(
            mesh, max_refinements, max_points, preview_depth=0
        )
    else:
        max_h_refinements = _bounded_density_bisections(
            mesh, max_h_refinements, max_points
        )
    key_indices = {key: index for index, key in enumerate(density_coordinates.keys)}
    components = np.asarray(
        [(key_indices[key], row, col) for key, row, col in density_coordinates.entries],
        dtype=np.int64,
    ).reshape((-1, 3))
    common = dict(
        mu=float(mu),
        lattice_vectors=density_coordinates.keys,
        components=components,
        target_error=float(density_atol),
        max_refinements=max_refinements,
    )
    with _integration_context(num_threads):
        if not prescribed:
            return mesh.integrate_density_components_p(
                **common,
                max_degree=max_degree,
                max_h_refinements=max_h_refinements,
            )
        return mesh.integrate_density_components(
            **common,
            preview_depth=0,
            min_refinement_batch_size=_MIN_REFINEMENT_BATCH_SIZE,
            max_refinement_batch_size=_MAX_REFINEMENT_BATCH_SIZE,
        )


@dataclass
class _Work:
    evaluations: int = 0
    diagonalizations: int = 0
    refinements: int = 0
    p_refinements: int = 0
    density_leaves: int | None = None
    charge_calls: int = 0
    density_calls: int = 0


class SimplexEvaluator:
    """One native mesh and the work performed while solving its density."""

    def __init__(self, problem: DensityProblem):
        self.problem = problem
        self.settings = problem.integration
        self.mesh = _spectral_mesh(
            problem.hamiltonian,
            nk=self.settings.nk
            if self.settings.nk is not None
            else self.settings.initial_nk,
            max_points=self.settings.max_points,
        )
        self.work = _Work()

    def remaining_refinements(self):
        limit = self.settings.max_refinements
        return None if limit is None else limit - self.work.refinements

    def charge(self, mu: float, *, adaptive: bool):
        self.work.charge_calls += 1
        if not adaptive:
            result, evaluations = _evaluate_charge(
                self.mesh, mu=mu, num_threads=self.settings.num_threads
            )
            self.work.evaluations += evaluations
            self.work.diagonalizations += evaluations
            return result
        result = _integrate_charge(
            self.mesh,
            mu=mu,
            charge_tol=self.problem.tolerances.charge_integration,
            max_refinements=self.remaining_refinements(),
            num_threads=self.settings.num_threads,
            max_points=self.settings.max_points,
        )
        if not result.stats.target_reached:
            raise RuntimeError(
                "Adaptive simplex loop did not converge while evaluating charge"
            )
        stats = result.error_stats
        self.work.evaluations += (
            result.stats.evaluations + stats.hamiltonian_evaluations
        )
        self.work.diagonalizations += (
            result.stats.evaluations
            + stats.full_eigensystems
            + stats.reduced_eigensystems
            + stats.norm_eigensystems
        )
        self.work.refinements += result.stats.refinements
        return result

    def density(self, mu: float, *, target_error: float | None = None):
        coordinates = self.problem.density_coordinates
        prescribed = self.settings.nk is not None
        if not coordinates.value_count:
            return _DensityEntries(
                coordinates, np.empty(0, complex), None if prescribed else np.empty(0)
            )
        density_target = (
            self.problem.tolerances.density_matrix_integration
            if target_error is None
            else target_error
        )
        result = _integrate_density(
            self.mesh,
            coordinates,
            mu=mu,
            density_atol=0.0 if prescribed else density_target,
            max_refinements=self.remaining_refinements()
            if prescribed
            else self.settings.max_refinements,
            num_threads=self.settings.num_threads,
            prescribed=prescribed,
            max_points=self.settings.max_points,
            max_degree=self.settings.density_max_degree,
            max_h_refinements=self.remaining_refinements() if not prescribed else 0,
        )
        if not result.stats.target_reached:
            raise RuntimeError(
                "Adaptive simplex loop did not converge while evaluating density"
            )
        self.work.density_calls += 1
        self.work.evaluations += result.stats.evaluations
        self.work.diagonalizations += result.stats.evaluations
        self.work.refinements += result.stats.refinements
        self.work.p_refinements += (
            int(result.stats.p_refinements) if not prescribed else 0
        )
        if not prescribed:
            self.work.density_leaves = int(result.stats.active_simplices)
        errors = (
            None
            if prescribed
            else np.full(coordinates.value_count, result.stopping_error)
        )
        return _DensityEntries(coordinates, result.values, errors)

    def density_trace(self, mu, density):
        """Read the density trace without refining the charge mesh.

        Selected entries may omit diagonal elements. For adaptive integration,
        their trace is the occupied volume on the frozen charge mesh.
        """
        trace = density.trace()
        if trace is not None:
            return trace
        if self.settings.nk is None:
            # Trace of an occupied projector is its occupation. The p rule
            # integrates this constant exactly on every charge-mesh simplex.
            return float(np.sum(self.mesh.occupied_weights(float(mu))))
        size = self.problem.density_coordinates.size
        local = (0,) * self.mesh.ndim
        diagonal = np.arange(size)
        coordinates = DensityCoordinates.from_pairs(
            size=size, keys=[local], pairs_by_key={local: (diagonal, diagonal)}
        )
        result = _integrate_density(
            self.mesh,
            coordinates,
            mu=mu,
            density_atol=1e100,
            max_refinements=0,
            num_threads=self.settings.num_threads,
            prescribed=self.settings.nk is not None,
        )
        self.work.density_calls += 1
        self.work.evaluations += result.stats.evaluations
        self.work.diagonalizations += result.stats.evaluations
        return float(np.sum(result.values).real)

    def statistics(self, charge_evaluations: int):
        work, mesh = self.work, self.mesh
        return IntegrationInfo(
            n_kernel_evals=int(work.evaluations),
            n_cached_nodes=int(mesh.cached_vertices),
            n_leaves=work.density_leaves
            if work.density_leaves is not None
            else int(mesh.active_simplices),
            refinements=int(work.refinements),
            p_refinements=int(work.p_refinements),
            error_estimate_available=self.settings.nk is None,
            num_threads=self.settings.num_threads,
            requested_nk=self.settings.nk,
            n_kpoints=int(mesh.active_vertices),
            n_diagonalizations=int(work.diagonalizations),
            charge_evaluations=charge_evaluations,
            charge_integration_calls=work.charge_calls,
            density_integration_calls=work.density_calls,
        )
