import numpy as np
import pytest
from scipy.optimize import brentq

from meanfi import AdaptiveQuadrature, LinearMixing, Model, solver, tb_to_kfunc
from meanfi.meanfield import bdg_correction_from_density
from meanfi.density.filling import charge_diagonal


pytestmark = [pytest.mark.physics, pytest.mark.perf_slow]


def _reference_density_keys(model: Model, guess) -> list[tuple[int, ...]]:
    del guess
    keys = list(model.h_int)
    onsite = (0,) * model._ndim
    if onsite not in keys:
        keys.append(onsite)
    return sorted(keys)


def _dense_bdg_density(model: Model, meanfield, *, keys, nk: int) -> tuple[float, dict]:
    hkfunc = tb_to_kfunc(model.bdg_hamiltonian_from_meanfield(meanfield))
    axis = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    if model._ndim == 1:
        points = axis[:, None]
    else:
        kx, ky = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack([kx.ravel(), ky.ravel()], axis=-1)

    hamiltonians = hkfunc(points)
    q_matrix = np.diag(charge_diagonal(model._ndof))

    def density_at_mu(mu: float) -> np.ndarray:
        shifted = hamiltonians - mu * q_matrix
        eigenvalues, eigenvectors = np.linalg.eigh(shifted)
        occupation = 1.0 / (np.exp(eigenvalues / model.kT) + 1.0)
        return (
            eigenvectors
            * occupation[..., np.newaxis, :]
            @ eigenvectors.conj().swapaxes(-1, -2)
        )

    def filling_residual(mu: float) -> float:
        density_k = density_at_mu(mu)
        electron_block = density_k[:, : model._ndof, : model._ndof]
        return float(
            np.mean(np.trace(electron_block, axis1=1, axis2=2).real) - model.filling
        )

    lower = -12.0
    upper = 12.0
    while filling_residual(lower) > 0.0:
        lower *= 2.0
    while filling_residual(upper) < 0.0:
        upper *= 2.0
    mu = float(brentq(filling_residual, lower, upper))

    density_k = density_at_mu(mu)
    density = {}
    for key in keys:
        phase = np.exp(1j * np.dot(points, np.asarray(key, dtype=float)))
        density[key] = np.einsum("k,kab->ab", phase / points.shape[0], density_k)
    return mu, density


def _dense_bdg_scf_reference(
    model: Model,
    guess,
    *,
    nk: int,
    alpha: float,
    tol: float,
    max_iterations: int = 200,
):
    keys = _reference_density_keys(model, guess)
    meanfield = {key: np.array(value, dtype=complex) for key, value in guess.items()}

    for _ in range(max_iterations):
        mu, density = _dense_bdg_density(model, meanfield, keys=keys, nk=nk)
        updated = bdg_correction_from_density(density, model)
        residual = max(
            np.max(np.abs(updated.get(key, 0.0) - meanfield.get(key, 0.0)))
            for key in frozenset(updated) | frozenset(meanfield)
        )
        if residual <= tol:
            return mu, updated
        meanfield = {
            key: np.asarray(
                meanfield.get(key, 0.0)
                + alpha * (updated.get(key, 0.0) - meanfield.get(key, 0.0)),
                dtype=complex,
            )
            for key in frozenset(updated) | frozenset(meanfield)
        }

    raise AssertionError("Dense BdG reference SCF did not converge")


def test_bdg_solver_matches_1d_nonlocal_odd_parity_reference():
    hopping = 1.0
    coupling = 1.8
    filling = 0.35
    kT = 0.08

    h_0 = {
        (0,): np.zeros((1, 1), dtype=complex),
        (1,): -hopping * np.eye(1, dtype=complex),
        (-1,): -hopping * np.eye(1, dtype=complex),
    }
    h_int = {
        (1,): coupling * np.ones((1, 1), dtype=complex),
        (-1,): coupling * np.ones((1, 1), dtype=complex),
    }
    guess = {
        (1,): np.array([[0.0, 0.25], [-0.25, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, -0.25], [0.25, 0.0]], dtype=complex),
    }

    model = Model(
        h_0,
        h_int,
        filling=filling,
        kT=kT,
        superconducting=True,
    )
    reference_mu, reference_meanfield = _dense_bdg_scf_reference(
        model,
        guess,
        nk=4001,
        alpha=0.6,
        tol=1e-8,
    )

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=2e-4,
            max_refinements=120,
        ),
        scf=LinearMixing(max_iterations=140, alpha=0.6),
        scf_tol=2e-4,
        filling_tol=2e-4,
    )

    assert result.info.residual_norm <= 2e-4
    assert abs(result.density_matrix_result.filling - filling) <= 2e-4
    assert result.mf[(1,)][0, 1].real * result.mf[(-1,)][0, 1].real < 0.0
    assert (
        abs(result.mf[(1,)][0, 1].real - reference_meanfield[(1,)][0, 1].real) <= 3e-3
    )
    assert (
        abs(result.mf[(-1,)][0, 1].real - reference_meanfield[(-1,)][0, 1].real) <= 3e-3
    )
    assert abs(result.density_matrix_result.mu - reference_mu) <= 2e-3

