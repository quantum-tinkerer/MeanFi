import numpy as np
import pytest
from scipy.optimize import brentq

from meanfi import AdaptiveQuadrature, LinearMixing, Model, solver, tb_to_kfunc
from meanfi.superconducting.bdg import bdg_correction_from_density, charge_diagonal


pytestmark = pytest.mark.physics


def _bdg_guess(delta: complex):
    return {(0, 0): np.array([[0.0, delta], [np.conj(delta), 0.0]], dtype=complex)}


def _local_gap_reference(*, coupling: float, kT: float) -> float:
    lower = 1e-12
    upper = coupling
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        residual = midpoint - 0.5 * coupling * np.tanh(midpoint / (2.0 * kT))
        if residual > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return float(0.5 * (lower + upper))


def _dispersive_bdg_reference_1d(
    *,
    hopping: float,
    coupling: float,
    filling: float,
    kT: float,
    nk: int,
) -> tuple[float, float]:
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    dispersion = -2.0 * hopping * np.cos(k)

    def filling_residual(mu: float, delta: float) -> float:
        xi = dispersion - mu
        energy = np.sqrt(xi**2 + delta**2)
        occupation = 0.5 * (1.0 - xi * np.tanh(energy / (2.0 * kT)) / energy)
        return float(np.mean(occupation) - filling)

    def chemical_potential(delta: float) -> float:
        lower = float(np.min(dispersion) - coupling - abs(delta) - 2.0)
        upper = float(np.max(dispersion) + coupling + abs(delta) + 2.0)
        return float(brentq(lambda mu: filling_residual(mu, delta), lower, upper))

    def gap_residual(delta: float) -> float:
        mu = chemical_potential(delta)
        xi = dispersion - mu
        energy = np.sqrt(xi**2 + delta**2)
        kernel = np.tanh(energy / (2.0 * kT)) / (2.0 * energy)
        return float(delta * (1.0 - coupling * np.mean(kernel)))

    lower = 1e-10
    upper = max(2.0 * coupling, 4.0 * hopping)
    assert gap_residual(lower) < 0.0
    assert gap_residual(upper) > 0.0
    delta = float(brentq(gap_residual, lower, upper))
    return delta, chemical_potential(delta)


def _dispersive_bdg_reference_2d(
    *,
    hopping: float,
    coupling: float,
    filling: float,
    kT: float,
    nk: int,
) -> tuple[float, float]:
    axis = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(axis, axis, indexing="ij")
    dispersion = -2.0 * hopping * (np.cos(kx) + np.cos(ky))

    def filling_residual(mu: float, delta: float) -> float:
        xi = dispersion - mu
        energy = np.sqrt(xi**2 + delta**2)
        occupation = 0.5 * (1.0 - xi * np.tanh(energy / (2.0 * kT)) / energy)
        return float(np.mean(occupation) - filling)

    def chemical_potential(delta: float) -> float:
        lower = float(np.min(dispersion) - coupling - abs(delta) - 2.0)
        upper = float(np.max(dispersion) + coupling + abs(delta) + 2.0)
        return float(brentq(lambda mu: filling_residual(mu, delta), lower, upper))

    def gap_residual(delta: float) -> float:
        mu = chemical_potential(delta)
        xi = dispersion - mu
        energy = np.sqrt(xi**2 + delta**2)
        kernel = np.tanh(energy / (2.0 * kT)) / (2.0 * energy)
        return float(delta * (1.0 - coupling * np.mean(kernel)))

    lower = 1e-10
    upper = max(2.0 * coupling, 4.0 * hopping)
    assert gap_residual(lower) < 0.0
    assert gap_residual(upper) > 0.0
    delta = float(brentq(gap_residual, lower, upper))
    return delta, chemical_potential(delta)


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
                meanfield.get(key, 0.0) + alpha * (updated.get(key, 0.0) - meanfield.get(key, 0.0)),
                dtype=complex,
            )
            for key in frozenset(updated) | frozenset(meanfield)
        }

    raise AssertionError("Dense BdG reference SCF did not converge")


def test_bdg_solver_matches_2d_local_gap_equation():
    coupling = 1.0
    kT = 0.2
    local = (0, 0)

    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[coupling]], dtype=complex)},
        filling=0.5,
        kT=kT,
        superconducting=True,
    )

    result = solver(
        model,
        _bdg_guess(0.3),
        integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
        scf=LinearMixing(max_iterations=80, alpha=0.8),
        scf_tol=1e-6,
        filling_tol=1e-6,
    )

    assert result.info.residual_norm <= 1e-6
    assert abs(result.density_matrix_result.filling - model.filling) <= 1e-6
    assert abs(result.mf[local][0, 1].real - _local_gap_reference(coupling=coupling, kT=kT)) <= 5e-5


def test_bdg_solver_matches_1d_dispersive_gap_and_number_equations():
    hopping = 1.0
    coupling = 2.0
    filling = 0.4
    kT = 0.05
    local = (0,)

    h_0 = {
        local: np.zeros((1, 1), dtype=complex),
        (1,): -hopping * np.eye(1, dtype=complex),
        (-1,): -hopping * np.eye(1, dtype=complex),
    }
    h_int = {local: coupling * np.ones((1, 1), dtype=complex)}
    guess = {local: np.array([[0.0, 0.3], [0.3, 0.0]], dtype=complex)}
    reference_delta, reference_mu = _dispersive_bdg_reference_1d(
        hopping=hopping,
        coupling=coupling,
        filling=filling,
        kT=kT,
        nk=20001,
    )

    model = Model(
        h_0,
        h_int,
        filling=filling,
        kT=kT,
        superconducting=True,
    )

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
        scf=LinearMixing(max_iterations=200, alpha=0.8),
        scf_tol=1e-6,
        filling_tol=1e-6,
    )

    assert result.info.residual_norm <= 1e-6
    assert abs(result.density_matrix_result.filling - filling) <= 1e-6
    assert abs(result.mf[local][0, 1].real - reference_delta) <= 5e-4
    assert abs(result.density_matrix_result.mu - reference_mu) <= 5e-4


def test_bdg_solver_matches_2d_dispersive_gap_and_number_equations():
    hopping = 0.5
    coupling = 2.0
    filling = 0.35
    kT = 0.1
    local = (0, 0)

    h_0 = {
        local: np.zeros((1, 1), dtype=complex),
        (1, 0): -hopping * np.eye(1, dtype=complex),
        (-1, 0): -hopping * np.eye(1, dtype=complex),
        (0, 1): -hopping * np.eye(1, dtype=complex),
        (0, -1): -hopping * np.eye(1, dtype=complex),
    }
    h_int = {local: coupling * np.ones((1, 1), dtype=complex)}
    guess = {local: np.array([[0.0, 0.2], [0.2, 0.0]], dtype=complex)}
    reference_delta, reference_mu = _dispersive_bdg_reference_2d(
        hopping=hopping,
        coupling=coupling,
        filling=filling,
        kT=kT,
        nk=501,
    )

    model = Model(
        h_0,
        h_int,
        filling=filling,
        kT=kT,
        superconducting=True,
    )

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=5e-4,
            max_refinements=200,
        ),
        scf=LinearMixing(max_iterations=120, alpha=0.8),
        scf_tol=5e-4,
        filling_tol=5e-4,
    )

    assert result.info.residual_norm <= 5e-4
    assert abs(result.density_matrix_result.filling - filling) <= 5e-4
    assert abs(result.mf[local][0, 1].real - reference_delta) <= 3e-3
    assert abs(result.density_matrix_result.mu - reference_mu) <= 2e-3


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
    assert abs(result.mf[(1,)][0, 1].real - reference_meanfield[(1,)][0, 1].real) <= 3e-3
    assert abs(result.mf[(-1,)][0, 1].real - reference_meanfield[(-1,)][0, 1].real) <= 3e-3
    assert abs(result.density_matrix_result.mu - reference_mu) <= 2e-3


def test_bdg_solver_matches_2d_d_wave_reference():
    hopping = 0.4
    coupling = 1.5
    filling = 0.5
    kT = 0.08

    h_0 = {
        (0, 0): np.zeros((1, 1), dtype=complex),
        (1, 0): -hopping * np.eye(1, dtype=complex),
        (-1, 0): -hopping * np.eye(1, dtype=complex),
        (0, 1): -hopping * np.eye(1, dtype=complex),
        (0, -1): -hopping * np.eye(1, dtype=complex),
    }
    h_int = {
        (1, 0): coupling * np.ones((1, 1), dtype=complex),
        (-1, 0): coupling * np.ones((1, 1), dtype=complex),
        (0, 1): coupling * np.ones((1, 1), dtype=complex),
        (0, -1): coupling * np.ones((1, 1), dtype=complex),
    }
    guess = {
        (1, 0): np.array([[0.0, 0.3], [0.3, 0.0]], dtype=complex),
        (-1, 0): np.array([[0.0, 0.3], [0.3, 0.0]], dtype=complex),
        (0, 1): np.array([[0.0, -0.3], [-0.3, 0.0]], dtype=complex),
        (0, -1): np.array([[0.0, -0.3], [-0.3, 0.0]], dtype=complex),
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
        nk=121,
        alpha=0.35,
        tol=1e-6,
    )

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=2e-4,
            max_refinements=220,
        ),
        scf=LinearMixing(max_iterations=140, alpha=0.35),
        scf_tol=2e-4,
        filling_tol=2e-4,
    )

    assert result.info.residual_norm <= 2e-4
    assert abs(result.density_matrix_result.filling - filling) <= 2e-4
    assert result.mf[(1, 0)][0, 1].real * result.mf[(0, 1)][0, 1].real < 0.0
    assert abs(result.mf[(1, 0)][0, 1].real - reference_meanfield[(1, 0)][0, 1].real) <= 3e-3
    assert abs(result.mf[(0, 1)][0, 1].real - reference_meanfield[(0, 1)][0, 1].real) <= 3e-3
    assert abs(result.mf[(-1, 0)][0, 1].real - result.mf[(1, 0)][0, 1].real) <= 2e-3
    assert abs(result.mf[(0, -1)][0, 1].real - result.mf[(0, 1)][0, 1].real) <= 2e-3
    assert abs(result.density_matrix_result.mu - reference_mu) <= 4e-3


def test_bdg_solver_matches_multi_orbital_onsite_pairing_reference():
    h_0 = {
        (0,): np.array([[0.2, 0.15], [0.15, -0.1]], dtype=complex),
        (1,): np.array([[-0.7, 0.05], [0.02, -0.4]], dtype=complex),
        (-1,): np.array([[-0.7, 0.02], [0.05, -0.4]], dtype=complex),
    }
    h_int = {(0,): np.array([[1.8, 0.4], [0.4, 1.3]], dtype=complex)}
    filling = 1.3
    kT = 0.09
    guess = {
        (0,): np.block(
            [
                [np.zeros((2, 2), dtype=complex), np.array([[0.25, 0.08], [0.08, 0.18]], dtype=complex)],
                [np.array([[0.25, 0.08], [0.08, 0.18]], dtype=complex), np.zeros((2, 2), dtype=complex)],
            ]
        )
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
        nk=2001,
        alpha=0.55,
        tol=1e-7,
    )

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=3e-4,
            max_refinements=160,
        ),
        scf=LinearMixing(max_iterations=140, alpha=0.55),
        scf_tol=3e-4,
        filling_tol=3e-4,
    )

    assert result.info.residual_norm <= 3e-4
    assert abs(result.density_matrix_result.filling - filling) <= 3e-4
    assert np.max(
        np.abs(result.mf[(0,)][:2, 2:] - reference_meanfield[(0,)][:2, 2:])
    ) <= 3e-3
    assert abs(result.density_matrix_result.mu - reference_mu) <= 2e-3


def test_bdg_solver_matches_frustrated_triangular_pairing_reference():
    hopping = 0.25
    coupling = 1.0
    filling = 0.55
    kT = 0.14
    bonds = [(1, 0), (0, 1), (-1, 1)]
    amplitudes = [1.0, -0.5, -0.5]

    h_0 = {(0, 0): np.zeros((1, 1), dtype=complex)}
    h_int = {}
    guess = {}
    for bond, amplitude in zip(bonds, amplitudes):
        opposite = tuple(-np.asarray(bond, dtype=int))
        h_0[bond] = -hopping * np.eye(1, dtype=complex)
        h_0[opposite] = -hopping * np.eye(1, dtype=complex)
        h_int[bond] = coupling * np.ones((1, 1), dtype=complex)
        h_int[opposite] = coupling * np.ones((1, 1), dtype=complex)
        guess[bond] = np.array([[0.0, 0.12 * amplitude], [0.12 * amplitude, 0.0]], dtype=complex)
        guess[opposite] = np.array(
            [[0.0, 0.12 * amplitude], [0.12 * amplitude, 0.0]],
            dtype=complex,
        )

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
        nk=81,
        alpha=0.25,
        tol=2e-6,
        max_iterations=240,
    )

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=4e-4,
            max_refinements=220,
        ),
        scf=LinearMixing(max_iterations=180, alpha=0.25),
        scf_tol=4e-4,
        filling_tol=4e-4,
    )

    assert result.info.residual_norm <= 4e-4
    assert abs(result.density_matrix_result.filling - filling) <= 4e-4
    assert result.mf[(1, 0)][0, 1].real > 0.0
    assert result.mf[(0, 1)][0, 1].real < 0.0
    assert result.mf[(-1, 1)][0, 1].real < 0.0
    assert abs(result.mf[(0, 1)][0, 1].real - result.mf[(-1, 1)][0, 1].real) <= 2e-3
    assert abs(result.mf[(1, 0)][0, 1].real - reference_meanfield[(1, 0)][0, 1].real) <= 8e-3
    assert abs(result.mf[(0, 1)][0, 1].real - reference_meanfield[(0, 1)][0, 1].real) <= 4e-3
    assert abs(result.density_matrix_result.mu - reference_mu) <= 8e-3
