from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meanfi.density.integrate.common import uniform_grid_info, wrap_density_result
from meanfi.density.kpoint.matrix_functions import (
    DirectDiagonalization,
    RationalFOE,
    resolve_matrix_function,
    selected_density_values_from_eigensystem,
    shift_by_mu,
)
from meanfi.density.kpoint.matrix_functions.rational import (
    PreparedMumpsRationalNode,
)
from meanfi.density.kpoint.occupations import fermi_dirac
from meanfi.density.integrate.methods import UniformGrid
from meanfi.density.integrate.workspace import workspace_complex_dtype
from meanfi.space.density_selection import DensitySelection
from meanfi.space.density_selection import full_density_selection
from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like, to_dense
from meanfi.tb.transforms import kgrid_to_tb, tb_to_kfunc, tb_to_kgrid
from meanfi.tb.validate import tb_dimension, tb_orbital_count


def _ifft_grid_index(
    key: tuple[int, ...],
    grid_shape: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(k % n for k, n in zip(key, grid_shape, strict=True))


def _selected_value_grid_to_tb(
    density_selection: DensitySelection,
    k_value_grid: np.ndarray,
    *,
    ndim: int,
) -> _tb_type:
    """Transform selected k-grid values into selected real-space TB blocks."""

    real_space_values = np.fft.ifftn(k_value_grid, axes=np.arange(ndim))
    grid_shape = tuple(real_space_values.shape[:ndim])
    rho: _tb_type = {}
    for selection in density_selection.key_selections:
        block = np.zeros(
            (density_selection.size, density_selection.size), dtype=complex
        )
        if selection.rows.size:
            key_values = np.asarray(
                real_space_values[_ifft_grid_index(selection.key, grid_shape)],
                dtype=complex,
            )
            block[selection.rows, selection.cols] = key_values[selection.value_slice]
        rho[selection.key] = block
    return rho


def uniform_grid_density_terms(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    nk: int,
    density_selection: DensitySelection | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> tuple[_tb_type, float]:
    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        matrix = np.asarray(hamiltonian[tuple()], dtype=workspace_dtype)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
        if density_selection is not None:
            values = density_selection.values_from_assembled_matrix(density)
            rho = density_selection.values_and_errors_to_tb(
                values,
                np.zeros(density_selection.value_count, dtype=float),
            )[0]
            return rho, float(np.sum(occupation))
        return {tuple(): density}, float(np.sum(occupation))

    kham = np.asarray(tb_to_kgrid(hamiltonian, nk=nk), dtype=workspace_dtype)
    eigenvalues, eigenvectors = np.linalg.eigh(kham)
    occupation = fermi_dirac(eigenvalues, kT, mu)
    filling = float(np.mean(np.sum(occupation, axis=-1)))
    if density_selection is None:
        occupied_vectors = eigenvectors * occupation[..., np.newaxis, :]
        density_matrix_k = occupied_vectors @ eigenvectors.conj().swapaxes(-1, -2)
        density_matrix = kgrid_to_tb(density_matrix_k)
        return density_matrix, filling

    value_grid = selected_density_values_from_eigensystem(
        eigenvectors,
        occupation,
        density_selection,
    )
    density_matrix = _selected_value_grid_to_tb(
        density_selection,
        value_grid,
        ndim=ndim,
    )
    return density_matrix, filling


def uniform_grid_fermi_level(eigenvalues: np.ndarray, filling: float) -> float:
    norbs = eigenvalues.shape[-1]
    flat = np.sort(eigenvalues.reshape(-1))
    n_total = flat.size
    index = int(round(n_total * filling / norbs))
    if index >= n_total:
        return float(flat[-1])
    if index == 0:
        return float(flat[0])
    return float((flat[index - 1] + flat[index]) / 2.0)


def uniform_grid_kpoints(ndim: int, nk: int) -> np.ndarray:
    if ndim == 0:
        return np.zeros((1, 0), dtype=float)
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    ks = np.concatenate((ks[nk // 2 :], ks[: nk // 2]), axis=0)
    grid = np.meshgrid(*([ks] * ndim), indexing="ij")
    return np.stack([axis.reshape(-1) for axis in grid], axis=-1)


def _tb_point_evaluator(
    hamiltonian: _tb_type,
    *,
    workspace_dtype: np.dtype,
):
    if not any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
        hkfunc = tb_to_kfunc(hamiltonian)

        def dense_point(kpoint: np.ndarray):
            return np.asarray(hkfunc(kpoint), dtype=workspace_dtype)

        return dense_point

    key_array = np.asarray(list(hamiltonian.keys()), dtype=float)
    matrices = [
        as_sparse(
            value.astype(workspace_dtype, copy=False)
            if hasattr(value, "astype")
            else value
        ).tocsr()
        for value in hamiltonian.values()
    ]

    def sparse_point(kpoint: np.ndarray):
        phases = np.exp(-1j * np.dot(key_array, np.asarray(kpoint, dtype=float)))
        result = None
        for phase, matrix in zip(phases, matrices, strict=True):
            term = matrix * complex(phase)
            result = term if result is None else result + term
        return result.tocsr()

    return sparse_point


@dataclass
class _PreparedDirectNode:
    matrix: np.ndarray
    kT: float
    q_diag: np.ndarray
    trace_weights_diag: np.ndarray
    density_selection: DensitySelection
    workspace_dtype: np.dtype

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=self.workspace_dtype)
        self.q_diag = np.asarray(self.q_diag, dtype=float)
        self.trace_weights_diag = np.asarray(self.trace_weights_diag, dtype=float)
        self._last_mu: float | None = None
        self._last_eigenvalues: np.ndarray | None = None
        self._last_eigenvectors: np.ndarray | None = None
        self._last_occupation: np.ndarray | None = None
        self._last_derivative_occupation: np.ndarray | None = None

    def _eigensystem(
        self, mu: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._last_mu == float(mu):
            assert self._last_eigenvalues is not None
            assert self._last_eigenvectors is not None
            assert self._last_occupation is not None
            assert self._last_derivative_occupation is not None
            return (
                self._last_eigenvalues,
                self._last_eigenvectors,
                self._last_occupation,
                self._last_derivative_occupation,
            )

        shifted = np.asarray(
            shift_by_mu(self.matrix, mu, self.q_diag, dtype=self.workspace_dtype),
            dtype=self.workspace_dtype,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(shifted)
        occupation = np.asarray(fermi_dirac(eigenvalues, self.kT, 0.0), dtype=float)
        if self.kT > 0:
            derivative_occupation = np.asarray(
                occupation * (1.0 - occupation) / self.kT,
                dtype=float,
            )
        else:
            derivative_occupation = np.zeros_like(occupation)
        self._last_mu = float(mu)
        self._last_eigenvalues = eigenvalues
        self._last_eigenvectors = eigenvectors
        self._last_occupation = occupation
        self._last_derivative_occupation = derivative_occupation
        return eigenvalues, eigenvectors, occupation, derivative_occupation

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        _eigenvalues, eigenvectors, occupation, derivative_occupation = (
            self._eigensystem(mu)
        )
        weights = np.abs(eigenvectors) ** 2
        diagonal = weights @ occupation
        derivative_diagonal = weights @ derivative_occupation
        charge = float(np.real(np.dot(self.trace_weights_diag, diagonal)))
        derivative = float(
            np.real(np.dot(self.trace_weights_diag, derivative_diagonal))
        )
        return charge, derivative

    def density_values_from_charge_order(self, mu: float) -> np.ndarray:
        _eigenvalues, eigenvectors, occupation, _derivative_occupation = (
            self._eigensystem(mu)
        )
        return selected_density_values_from_eigensystem(
            eigenvectors,
            occupation,
            self.density_selection,
        )


@dataclass(frozen=True)
class UniformGridNodeBundle:
    nodes: tuple[Any, ...]
    density_selection: DensitySelection
    grid_shape: tuple[int, ...]
    use_derivative: bool


def build_uniform_grid_node_bundle(
    hamiltonian: _tb_type,
    *,
    kT: float,
    nk: int,
    keys: list[tuple[int, ...]],
    matrix_function: DirectDiagonalization | RationalFOE,
    q_diag: np.ndarray,
    trace_weights_diag: np.ndarray,
    charge_tolerance: float,
    density_tolerance: float,
    density_selection: DensitySelection | None,
    workspace_dtype: np.dtype,
) -> UniformGridNodeBundle:
    ndim = tb_dimension(hamiltonian)
    size = int(q_diag.size)
    density_selection = (
        density_selection
        if density_selection is not None
        else full_density_selection(keys, size=size)
    )
    kpoints = uniform_grid_kpoints(ndim, nk)
    point_matrix = _tb_point_evaluator(hamiltonian, workspace_dtype=workspace_dtype)
    shared_aaa_interval_cache = [] if isinstance(matrix_function, RationalFOE) else None
    nodes: list[Any] = []
    use_derivative = kT > 0

    for kpoint in kpoints:
        matrix = point_matrix(kpoint)
        if isinstance(matrix_function, DirectDiagonalization):
            nodes.append(
                _PreparedDirectNode(
                    matrix=np.asarray(to_dense(matrix), dtype=workspace_dtype),
                    kT=kT,
                    q_diag=q_diag,
                    trace_weights_diag=trace_weights_diag,
                    density_selection=density_selection,
                    workspace_dtype=workspace_dtype,
                )
            )
            continue

        if is_sparse_like(matrix):
            nodes.append(
                PreparedMumpsRationalNode(
                    matrix,
                    kT=kT,
                    q_diag=q_diag,
                    options=matrix_function,
                    charge_tolerance=charge_tolerance,
                    density_selection=density_selection,
                    density_tolerance=density_tolerance,
                    workspace_dtype=workspace_dtype,
                    trace_weights_diag=trace_weights_diag,
                    shared_aaa_interval_cache=shared_aaa_interval_cache,
                )
            )
            use_derivative = False
            continue

        raise ValueError(
            "UniformGrid RationalFOE is supported only for sparse matrices"
        )

    return UniformGridNodeBundle(
        nodes=tuple(nodes),
        density_selection=density_selection,
        grid_shape=tuple([nk] * ndim),
        use_derivative=use_derivative,
    )


def uniform_grid_density_from_nodes(
    bundle: UniformGridNodeBundle,
    *,
    mu: float,
) -> tuple[_tb_type, float]:
    estimate = np.zeros(bundle.density_selection.value_count, dtype=complex)
    value_grid = (
        np.empty(
            (len(bundle.nodes), bundle.density_selection.value_count), dtype=complex
        )
        if bundle.grid_shape
        else None
    )
    charge = 0.0
    n_kpoints = len(bundle.nodes)
    for index, node in enumerate(bundle.nodes):
        node_charge, _node_derivative = node.charge_and_derivative(mu)
        charge += float(node_charge)
        packed = node.density_values_from_charge_order(mu)
        if value_grid is None:
            estimate = packed
        else:
            value_grid[index] = packed
    if n_kpoints:
        charge /= float(n_kpoints)
    if value_grid is None:
        density_matrix = bundle.density_selection.values_and_errors_to_tb(
            estimate,
            np.zeros(bundle.density_selection.value_count, dtype=float),
        )[0]
    else:
        grid = value_grid.reshape(
            bundle.grid_shape + (bundle.density_selection.value_count,)
        )
        density_matrix = _selected_value_grid_to_tb(
            bundle.density_selection,
            grid,
            ndim=len(bundle.grid_shape),
        )
    return density_matrix, charge


__all__ = [
    "UniformGridNodeBundle",
    "build_uniform_grid_node_bundle",
    "resolve_uniform_grid_matrix_function",
    "solve_uniform_grid_at_mu",
    "uniform_grid_density_from_nodes",
    "uniform_grid_density_terms",
    "uniform_grid_fermi_level",
    "uniform_grid_kpoints",
]


def _uniform_charge_weights(hamiltonian: _tb_type) -> np.ndarray:
    return np.ones(tb_orbital_count(hamiltonian), dtype=float)


def resolve_uniform_grid_matrix_function(
    selected: object | None,
    hamiltonian: _tb_type,
    *,
    kT: float,
) -> DirectDiagonalization | RationalFOE:
    if selected is None:
        if kT > 0 and any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
            return RationalFOE(rational_scheme="aaa")
        return DirectDiagonalization()
    resolved = resolve_matrix_function(selected)
    if not isinstance(resolved, (DirectDiagonalization, RationalFOE)):
        raise TypeError(
            "UniformGrid.matrix_function must be DirectDiagonalization or RationalFOE"
        )
    if isinstance(resolved, RationalFOE) and kT <= 0:
        raise ValueError("UniformGrid RationalFOE requires kT > 0")
    if isinstance(resolved, RationalFOE) and not any(
        is_sparse_like(matrix) for matrix in hamiltonian.values()
    ):
        raise ValueError(
            "UniformGrid RationalFOE is supported only for sparse matrices"
        )
    return resolved


def solve_uniform_grid_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: UniformGrid,
    density_selection: DensitySelection | None = None,
):
    ndim = tb_dimension(hamiltonian)
    workspace_dtype = workspace_complex_dtype(integration)
    matrix_function = resolve_uniform_grid_matrix_function(
        getattr(integration, "matrix_function", None),
        hamiltonian,
        kT=kT,
    )
    if isinstance(matrix_function, DirectDiagonalization) and not any(
        is_sparse_like(matrix) for matrix in hamiltonian.values()
    ):
        density_matrix, filling = uniform_grid_density_terms(
            hamiltonian,
            mu=mu,
            kT=kT,
            nk=integration.nk,
            density_selection=density_selection,
            workspace_dtype=workspace_dtype,
        )
    else:
        bundle = build_uniform_grid_node_bundle(
            hamiltonian,
            kT=kT,
            nk=integration.nk,
            keys=keys,
            matrix_function=matrix_function,
            q_diag=_uniform_charge_weights(hamiltonian),
            trace_weights_diag=_uniform_charge_weights(hamiltonian),
            charge_tolerance=integration.density_matrix_tol,
            density_tolerance=integration.density_matrix_tol,
            density_selection=density_selection,
            workspace_dtype=workspace_dtype,
        )
        density_matrix, filling = uniform_grid_density_from_nodes(bundle, mu=mu)

    return wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=None,
        mu=mu,
        filling=filling,
        target_filling=None,
        integration=integration,
        info=uniform_grid_info(
            integration=integration,
            hamiltonian=hamiltonian,
            n_kernel_evals=(integration.nk**ndim if ndim > 0 else 1),
            n_evaluator_evals=(integration.nk**ndim if ndim > 0 else 1),
            density_integration_calls=1,
            error_estimate_available=False,
        ),
        keys=keys,
    )
