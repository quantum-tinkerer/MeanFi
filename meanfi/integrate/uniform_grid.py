from __future__ import annotations

import numpy as np

from meanfi.core.validation import tb_dimension
from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import kgrid_to_tb, tb_to_kgrid

from .common import uniform_grid_info, wrap_density_result
from .methods import UniformGrid
from .occupations import fermi_dirac


def uniform_grid_density_terms(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    nk: int,
) -> tuple[_tb_type, float]:
    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian[tuple()])
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
        return {tuple(): density}, float(np.sum(occupation))

    kham = tb_to_kgrid(hamiltonian, nk=nk)
    eigenvalues, eigenvectors = np.linalg.eigh(kham)
    occupation = fermi_dirac(eigenvalues, kT, mu)
    occupied_vectors = eigenvectors * occupation[..., np.newaxis, :]
    density_matrix_k = occupied_vectors @ eigenvectors.conj().swapaxes(-1, -2)
    density_matrix = kgrid_to_tb(density_matrix_k)
    filling = float(np.mean(np.sum(occupation, axis=-1)))
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
):
    density_matrix, filling = uniform_grid_density_terms(
        hamiltonian,
        mu=mu,
        kT=kT,
        nk=integration.nk,
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
