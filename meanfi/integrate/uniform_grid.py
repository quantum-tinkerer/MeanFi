from __future__ import annotations

import numpy as np

from meanfi.core.validation import tb_dimension
from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import kgrid_to_tb, tb_to_kgrid

from .common import uniform_grid_info, wrap_density_result
from .density_support import DensityEntrySupport, workspace_complex_dtype
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


def solve_uniform_grid_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: UniformGrid,
    density_entry_support: DensityEntrySupport | None = None,
):
    density_matrix, filling = uniform_grid_density_terms(
        hamiltonian,
        mu=mu,
        kT=kT,
        nk=integration.nk,
        density_entry_support=None,
        workspace_dtype=workspace_complex_dtype(integration),
    )
    return wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=None,
        mu=mu,
        filling=filling,
        target_filling=None,
        integration=integration,
        info=uniform_grid_info(integration=integration, hamiltonian=hamiltonian),
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
    density_entry_support: DensityEntrySupport | None = None,
):
    del kT
    if filling_tol is not None or mu_tol != 1e-10 or max_mu_iterations is not None:
        raise ValueError(
            "UniformGrid does not support filling_tol, mu_tol, or max_mu_iterations"
        )

    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        eigenvalues = np.linalg.eigvalsh(hamiltonian[tuple()])
    else:
        kham = tb_to_kgrid(hamiltonian, nk=integration.nk)
        eigenvalues = np.linalg.eigvalsh(kham)
    mu = uniform_grid_fermi_level(eigenvalues, filling)
    density_matrix, resolved_filling = uniform_grid_density_terms(
        hamiltonian,
        mu=mu,
        kT=0.0,
        nk=integration.nk,
        density_entry_support=None,
        workspace_dtype=workspace_complex_dtype(integration),
    )
    return wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=None,
        mu=mu,
        filling=resolved_filling,
        target_filling=filling,
        integration=integration,
        info=uniform_grid_info(integration=integration, hamiltonian=hamiltonian),
        keys=keys,
    )
