from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from meanfi.integrate.filling import mu_bracket
from meanfi.tb.ops import as_sparse, is_sparse_like, to_dense
from meanfi.tb.validate import tb_dimension, tb_orbital_count
from meanfi.tb.ops import _tb_type
from meanfi.tb.transforms import kgrid_to_tb, tb_to_kfunc, tb_to_kgrid

from .common import uniform_grid_info, wrap_density_result
from meanfi.state.support import (
    DensityEntrySupport,
    full_density_entry_support,
    workspace_complex_dtype,
)
from .fixed_filling import solve_fixed_filling_root
from .matrix_functions import (
    DirectDiagonalization,
    RationalFOE,
    resolve_matrix_function,
    shift_by_mu,
)
from .matrix_functions.rational import PreparedMumpsRationalNode, PreparedRationalNode
from .methods import UniformGrid
from .occupations import fermi_dirac


def uniform_grid_density_terms(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    nk: int,
    density_entry_support: DensityEntrySupport | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> tuple[_tb_type, float]:
    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        matrix = np.asarray(hamiltonian[tuple()], dtype=workspace_dtype)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
        if density_entry_support is not None:
            values = density_entry_support.pack_columns(density)
            rho = density_entry_support.expand_entries(
                values,
                np.zeros(density_entry_support.output_size, dtype=float),
            )[0]
            return rho, float(np.sum(occupation))
        return {tuple(): density}, float(np.sum(occupation))

    kham = np.asarray(tb_to_kgrid(hamiltonian, nk=nk), dtype=workspace_dtype)
    eigenvalues, eigenvectors = np.linalg.eigh(kham)
    occupation = fermi_dirac(eigenvalues, kT, mu)
    filling = float(np.mean(np.sum(occupation, axis=-1)))
    if density_entry_support is None:
        occupied_vectors = eigenvectors * occupation[..., np.newaxis, :]
        density_matrix_k = occupied_vectors @ eigenvectors.conj().swapaxes(-1, -2)
        density_matrix = kgrid_to_tb(density_matrix_k)
        return density_matrix, filling

    selected_rows = np.take(eigenvectors.conj(), density_entry_support.selected_columns, axis=-2)
    density_columns = eigenvectors @ np.swapaxes(
        occupation[..., np.newaxis, :] * selected_rows,
        -1,
        -2,
    )
    entry_grid = np.empty(
        density_columns.shape[:-2] + (density_entry_support.output_size,),
        dtype=complex,
    )
    for index, (rows, positions) in enumerate(
        zip(density_entry_support.row_indices, density_entry_support.column_positions, strict=True)
    ):
        start = density_entry_support.offsets[index]
        stop = density_entry_support.offsets[index + 1]
        if stop == start:
            continue
        entry_grid[..., start:stop] = density_columns[..., rows, positions]
    density_matrix = density_entry_support.expand_ifft_entries(
        np.fft.ifftn(entry_grid, axes=np.arange(ndim))
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
        raise TypeError("UniformGrid.matrix_function must be DirectDiagonalization or RationalFOE")
    if isinstance(resolved, RationalFOE) and kT <= 0:
        raise ValueError("UniformGrid RationalFOE requires kT > 0")
    return resolved


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
        as_sparse(value.astype(workspace_dtype, copy=False) if hasattr(value, "astype") else value).tocsr()
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

    def _eigensystem(self, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        _eigenvalues, eigenvectors, occupation, derivative_occupation = self._eigensystem(mu)
        weights = np.abs(eigenvectors) ** 2
        diagonal = weights @ occupation
        derivative_diagonal = weights @ derivative_occupation
        charge = float(np.real(np.dot(self.trace_weights_diag, diagonal)))
        derivative = float(np.real(np.dot(self.trace_weights_diag, derivative_diagonal)))
        return charge, derivative

    def density_columns_from_charge_order(self, mu: float, basis: np.ndarray) -> np.ndarray:
        _eigenvalues, eigenvectors, occupation, _derivative_occupation = self._eigensystem(mu)
        overlap = eigenvectors.conj().T @ np.asarray(basis, dtype=self.workspace_dtype)
        return (eigenvectors * occupation[np.newaxis, :]) @ overlap


@dataclass(frozen=True)
class UniformGridNodeBundle:
    nodes: tuple[Any, ...]
    density_support: DensityEntrySupport
    basis: np.ndarray
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
    density_entry_support: DensityEntrySupport | None,
    workspace_dtype: np.dtype,
) -> UniformGridNodeBundle:
    ndim = tb_dimension(hamiltonian)
    size = int(q_diag.size)
    density_support = (
        density_entry_support
        if density_entry_support is not None
        else full_density_entry_support(keys, size=size)
    )
    basis = density_support.basis_block(dtype=workspace_dtype)
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
                    density_support=density_support,
                    density_tolerance=density_tolerance,
                    workspace_dtype=workspace_dtype,
                    trace_weights_diag=trace_weights_diag,
                    shared_aaa_interval_cache=shared_aaa_interval_cache,
                )
            )
            use_derivative = False
            continue

        nodes.append(
            PreparedRationalNode(
                np.asarray(matrix, dtype=workspace_dtype),
                kT=kT,
                q_diag=q_diag,
                options=matrix_function,
                charge_tolerance=charge_tolerance,
                workspace_dtype=workspace_dtype,
                trace_weights_diag=trace_weights_diag,
            )
        )
        if matrix_function.rational_scheme == "aaa":
            use_derivative = False

    return UniformGridNodeBundle(
        nodes=tuple(nodes),
        density_support=density_support,
        basis=basis,
        grid_shape=tuple([nk] * ndim),
        use_derivative=use_derivative,
    )


def uniform_grid_density_from_nodes(
    bundle: UniformGridNodeBundle,
    *,
    mu: float,
) -> tuple[_tb_type, float]:
    estimate = np.zeros(bundle.density_support.output_size, dtype=complex)
    entry_grid = (
        np.empty((len(bundle.nodes), bundle.density_support.output_size), dtype=complex)
        if bundle.grid_shape
        else None
    )
    charge = 0.0
    n_kpoints = len(bundle.nodes)
    for index, node in enumerate(bundle.nodes):
        node_charge, _node_derivative = node.charge_and_derivative(mu)
        charge += float(node_charge)
        density_columns = node.density_columns_from_charge_order(mu, bundle.basis)
        packed = bundle.density_support.pack_columns(density_columns)
        if entry_grid is None:
            estimate = packed
        else:
            entry_grid[index] = packed
    if n_kpoints:
        charge /= float(n_kpoints)
    if entry_grid is None:
        density_matrix = bundle.density_support.expand_entries(
            estimate,
            np.zeros(bundle.density_support.output_size, dtype=float),
        )[0]
    else:
        grid = entry_grid.reshape(bundle.grid_shape + (bundle.density_support.output_size,))
        density_matrix = bundle.density_support.expand_ifft_entries(
            np.fft.ifftn(grid, axes=np.arange(len(bundle.grid_shape)))
        )
    return density_matrix, charge


def uniform_grid_fixed_filling_from_nodes(
    bundle: UniformGridNodeBundle,
    *,
    hamiltonian: _tb_type,
    integration: UniformGrid,
    filling: float,
    mu_guess: float,
    filling_tol: float,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_bracket_builder,
) -> tuple[_tb_type, float, float, Any]:
    charge_calls = 0
    n_kpoints = len(bundle.nodes)

    def evaluate_charge(mu: float) -> tuple[float, float, float | None]:
        nonlocal charge_calls
        charge_calls += 1
        total_charge = 0.0
        total_derivative = 0.0
        for node in bundle.nodes:
            node_charge, node_derivative = node.charge_and_derivative(mu)
            total_charge += float(node_charge)
            total_derivative += float(node_derivative)
        if n_kpoints:
            total_charge /= float(n_kpoints)
            total_derivative /= float(n_kpoints)
        return total_charge, 0.0, (total_derivative if bundle.use_derivative else None)

    root = solve_fixed_filling_root(
        evaluate_charge=evaluate_charge,
        mu_bracket=mu_bracket_builder,
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        use_derivative=bundle.use_derivative,
    )
    density_matrix, resolved_filling = uniform_grid_density_from_nodes(bundle, mu=root.mu)
    info = uniform_grid_info(
        integration=integration,
        hamiltonian=hamiltonian,
        n_kernel_evals=(charge_calls + 1) * max(1, n_kpoints),
        n_evaluator_evals=(charge_calls + 1) * max(1, n_kpoints),
        root_iterations=root.root_iterations,
        charge_integration_calls=charge_calls,
        density_integration_calls=1,
        error_estimate_available=False,
    )
    return density_matrix, root.mu, resolved_filling, info


def solve_uniform_grid_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: UniformGrid,
    density_entry_support: DensityEntrySupport | None = None,
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
            density_entry_support=density_entry_support,
            workspace_dtype=workspace_dtype,
        )
    else:
        bundle = build_uniform_grid_node_bundle(
            hamiltonian,
            kT=kT,
            nk=integration.nk,
            keys=keys,
            matrix_function=matrix_function,
            q_diag=np.ones(tb_orbital_count(hamiltonian), dtype=float),
            trace_weights_diag=np.ones(tb_orbital_count(hamiltonian), dtype=float),
            charge_tolerance=integration.density_matrix_tol,
            density_tolerance=integration.density_matrix_tol,
            density_entry_support=density_entry_support,
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


def solve_uniform_grid_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: UniformGrid,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float = 0.0,
    density_entry_support: DensityEntrySupport | None = None,
):
    workspace_dtype = workspace_complex_dtype(integration)
    matrix_function = resolve_uniform_grid_matrix_function(
        getattr(integration, "matrix_function", None),
        hamiltonian,
        kT=kT,
    )
    if kT == 0 and isinstance(matrix_function, DirectDiagonalization):
        if any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
            point_matrix = _tb_point_evaluator(hamiltonian, workspace_dtype=workspace_dtype)
            eigenvalues = np.stack(
                [
                    np.linalg.eigvalsh(np.asarray(to_dense(point_matrix(kpoint)), dtype=workspace_dtype))
                    for kpoint in uniform_grid_kpoints(tb_dimension(hamiltonian), integration.nk)
                ],
                axis=0,
            )
        else:
            kham = tb_to_kgrid(hamiltonian, nk=integration.nk)
            eigenvalues = np.linalg.eigvalsh(kham)
        mu = uniform_grid_fermi_level(eigenvalues, filling)
        if any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
            bundle = build_uniform_grid_node_bundle(
                hamiltonian,
                kT=kT,
                nk=integration.nk,
                keys=keys,
                matrix_function=matrix_function,
                q_diag=np.ones(tb_orbital_count(hamiltonian), dtype=float),
                trace_weights_diag=np.ones(tb_orbital_count(hamiltonian), dtype=float),
                charge_tolerance=integration.density_matrix_tol,
                density_tolerance=integration.density_matrix_tol,
                density_entry_support=density_entry_support,
                workspace_dtype=workspace_dtype,
            )
            density_matrix, resolved_filling = uniform_grid_density_from_nodes(bundle, mu=mu)
        else:
            density_matrix, resolved_filling = uniform_grid_density_terms(
                hamiltonian,
                mu=mu,
                kT=0.0,
                nk=integration.nk,
                density_entry_support=density_entry_support,
                workspace_dtype=workspace_dtype,
            )
        info = uniform_grid_info(
            integration=integration,
            hamiltonian=hamiltonian,
            charge_integration_calls=1,
            density_integration_calls=1,
            root_iterations=1,
            error_estimate_available=False,
        )
        return wrap_density_result(
            density_matrix=density_matrix,
            density_matrix_error=None,
            mu=mu,
            filling=resolved_filling,
            target_filling=filling,
            integration=integration,
            info=info,
            keys=keys,
        )

    bundle = build_uniform_grid_node_bundle(
        hamiltonian,
        kT=kT,
        nk=integration.nk,
        keys=keys,
        matrix_function=matrix_function,
        q_diag=np.ones(tb_orbital_count(hamiltonian), dtype=float),
        trace_weights_diag=np.ones(tb_orbital_count(hamiltonian), dtype=float),
        charge_tolerance=(
            integration.density_matrix_tol if filling_tol is None else float(filling_tol)
        ),
        density_tolerance=integration.density_matrix_tol,
        density_entry_support=density_entry_support,
        workspace_dtype=workspace_dtype,
    )
    resolved_filling_tol = (
        float(tb_orbital_count(hamiltonian) * integration.density_matrix_tol)
        if filling_tol is None
        else float(filling_tol)
    )
    density_matrix, mu, resolved_filling, info = uniform_grid_fixed_filling_from_nodes(
        bundle,
        hamiltonian=hamiltonian,
        integration=integration,
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=resolved_filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_bracket_builder=lambda: mu_bracket(hamiltonian, kT),
    )
    return wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=None,
        mu=mu,
        filling=resolved_filling,
        target_filling=filling,
        integration=integration,
        info=info,
        keys=keys,
    )
