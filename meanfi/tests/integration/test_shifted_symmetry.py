"""Shifted real-space constraints agree with Bloch-space covariance."""

from dataclasses import replace

import numpy as np
import pytest

import meanfi as mf

pytestmark = pytest.mark.integration


@pytest.fixture
def glide_model():
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    forward = np.array([[0.0, 1.0], [0.0, 0.0]])
    transverse = -0.25 * np.eye(2) + 0.45j * np.diag([1, -1])
    h0 = {
        (0, 0): swap,
        (1, 0): forward,
        (-1, 0): forward.T,
        (0, 1): transverse,
        (0, -1): transverse.conj().T,
    }
    interaction = {(0, 0): 3 * swap, (1, 0): 3 * forward, (-1, 0): 3 * forward.T}
    glide = mf.SpatialSymmetry(np.diag([1, -1]), {(0, 0): forward.T, (1, 0): forward})
    return mf.Model(h0, interaction, 1.0, kT=0.3, spatial_symmetries=(glide,))


def glide_error(model, correction):
    points = np.random.default_rng(2).uniform(-np.pi, np.pi, (31, 2))
    forward = np.array([[0.0, 1.0], [0.0, 0.0]])
    unitary = forward.T + np.exp(-1j * points[:, 0, None, None]) * forward
    if model.superconducting:
        # U_hole(k) = U_electron(-k)*; the real shift blocks give the same U(k).
        nambu = np.zeros((len(points), 4, 4), complex)
        nambu[:, :2, :2] = nambu[:, 2:, 2:] = unitary
        unitary = nambu
    h = mf.tb_to_kfunc(model.hamiltonian_from_meanfield(correction))
    transformed = unitary @ h(points * [1, -1]) @ unitary.conj().transpose(0, 2, 1)
    return float(np.max(np.abs(h(points) - transformed)))


@pytest.mark.parametrize("superconducting", [False, True])
def test_shifted_constraints_preserve_dense_reference(glide_model, superconducting):
    model = replace(glide_model, superconducting=superconducting)
    unrestricted = replace(model, spatial_symmetries=())
    density = mf.density_matrix_at_mu(
        unrestricted, 0.17, keys=list(model.h_0), integration=mf.UniformGrid(nk=32**2)
    )
    expected = unrestricted.mean_field(density)
    constrained = model.mean_field(density)
    for key in expected:
        error = np.max(np.abs(expected[key] - constrained[key]))
        assert error < 1e-12, (
            f"Glide projection changed invariant dense density: {error}"
        )
    assert glide_error(model, model.random_meanfield(rng=4)) < 1e-12


def test_glide_scf_has_physical_symmetry_and_mesh_convergence(glide_model):
    scf = mf.EnergyDIIS()
    free = replace(glide_model, spatial_symmetries=())
    results = []
    for model in (free, glide_model):
        coarse = mf.solver(
            model,
            model.random_meanfield(rng=10, scale=0.05),
            scf=scf,
            integration=mf.UniformGrid(nk=32**2),
            tol=1e-7,
        )
        fine = mf.solver(
            model,
            coarse.mean_field,
            scf=scf,
            integration=mf.UniformGrid(nk=64**2),
            tol=1e-7,
            compute_free_energy=True,
        )
        error = np.max(np.abs(fine.density.values - coarse.density.values))
        assert error < 2e-6, f"Glide tutorial density mesh error: {error}"
        results.append(fine)
    assert glide_error(free, results[0].mean_field) > 1e-2
    assert glide_error(glide_model, results[1].mean_field) < 1e-12
    assert results[0].free_energy < results[1].free_energy
