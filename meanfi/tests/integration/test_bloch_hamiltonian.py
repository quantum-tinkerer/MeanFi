"""Shared callable/TB workflows and an analytic mapped-domain reference."""

from dataclasses import FrozenInstanceError, replace
from functools import partial

import numpy as np
import pytest

import meanfi as mf


pytestmark = pytest.mark.integration


def periodic_hamiltonian():
    hop = np.array([[0.5, 0.12j], [0.2, -0.1]])
    return {(0,): np.array([[-3, 0.2j], [-0.2j, 1]]), (1,): hop, (-1,): hop.conj().T}


@pytest.mark.parametrize("kT", [0.0, 0.2])
@pytest.mark.parametrize("prescribed", [False, True])
def test_callable_matches_tb_density_fourier_moments_and_energy(kT, prescribed):
    tb = periodic_hamiltonian()
    kfunc = mf.tb_to_kfunc(tb)
    callable_h = mf.BlochHamiltonian(lambda kx: kfunc([kx]))
    integration = (mf.FermiSimplex if kT == 0 else mf.UniformGrid)(
        nk=129 if prescribed else None
    )
    results = [
        mf.density_matrix(
            h,
            filling=0.7,
            kT=kT,
            keys=[(0,), (1,), (-1,)],
            integration=integration,
            tol=replace(
                mf.default_solver_tolerances(1e-5), filling_residual=1e-11, mu_tol=1e-12
            ),
            compute_free_energy=True,
        )
        for h in (tb, callable_h)
    ]
    # Identical physical integrands, including complex nonlocal Fourier moments.
    np.testing.assert_allclose(results[0].values, results[1].values, atol=2e-7, rtol=0)
    assert results[0].mu == pytest.approx(results[1].mu, abs=2e-6)
    assert results[0].band_energy == pytest.approx(results[1].band_energy, abs=2e-7)
    assert results[0].entropy == pytest.approx(results[1].entropy, abs=2e-7)


@pytest.mark.parametrize("bilinear", [False, True])
@pytest.mark.parametrize("kT", [0.0, 0.2])
def test_callable_scf_uses_shared_ediis_and_reference_energy(bilinear, kT):
    h0 = np.array([[-0.8, 0.1j], [-0.1j, 0.6]])
    reference = {(0,): np.array([[0.4, 0.03j], [-0.03j, 0.6]])}
    interaction = (
        mf.BilinearInteraction([mf.BilinearTerm(0.2, np.diag([1, 0]), np.diag([0, 1]))])
        if bilinear
        else {(0,): np.array([[0, 0.2], [0.2, 0]])}
    )
    tb_model = mf.Model({(0,): h0}, interaction, 1, kT=kT, reference=reference)
    model = replace(tb_model, h_0=mf.BlochHamiltonian(lambda kx: h0))
    # Deliberately off SCF: energy must subtract the INPUT correction in full.
    correction = {(0,): np.array([[0.9, 0.25j], [-0.25j, -0.4]])}
    density = mf.density_matrix(model, mean_field=correction, keys=[(0,)], tol=1e-8)
    exact = mf.evaluate_internal_energy(tb_model, density)
    assert density.internal_energy == pytest.approx(exact, abs=1e-12)
    result = mf.solver(
        model,
        model.random_meanfield(rng=4, scale=0.03),
        tol=1e-8,
        compute_free_energy=True,
    )
    assert result.converged
    full = mf.density_matrix(model, mean_field=result.mean_field, keys=[(0,)], tol=1e-8)
    assert result.internal_energy == pytest.approx(
        mf.evaluate_internal_energy(tb_model, full), abs=1e-12
    )
    assert result.free_energy == pytest.approx(
        result.internal_energy - kT * result.entropy, abs=1e-13
    )
    np.testing.assert_allclose(
        result.mean_field[(0,)], model.mean_field(result.density)[(0,)], atol=1e-8
    )
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        mf.evaluate_internal_energy(model, density)


def test_callable_with_nonlocal_density_density_correction():
    tb = periodic_hamiltonian()
    interaction = {
        (0,): np.zeros((2, 2)),
        (1,): np.full((2, 2), 0.2),
        (-1,): np.full((2, 2), 0.2),
    }
    kfunc = mf.tb_to_kfunc(tb)
    model = mf.Model(
        mf.BlochHamiltonian(lambda kx: kfunc([kx])), interaction, 0.7, kT=0.2
    )
    legacy = replace(model, h_0=tb)
    correction = model.random_meanfield(rng=2, scale=0.1)
    callable_h = model.hamiltonian_from_meanfield(correction)
    tb_h = mf.tb_to_kfunc(legacy.hamiltonian_from_meanfield(correction))
    for k in ([0.3], [2.1], [6.0]):
        np.testing.assert_allclose(callable_h(*k), tb_h(k), atol=1e-14)
    density = mf.density_matrix(model, mean_field=correction, keys=list(tb), tol=1e-7)
    assert density.internal_energy == pytest.approx(
        mf.evaluate_internal_energy(legacy, density), abs=1e-12
    )


def test_disk_map_against_analytic_dirac_density():
    mass, filling, cutoff = 0.2, 1.15, 0.16

    def disk_hamiltonian(kx, ky):
        r, theta = cutoff * np.sqrt(kx / (2 * np.pi)), ky
        z = r * np.exp(1j * theta) / cutoff
        return np.array([[mass, z.conjugate()], [z, -mass]])

    h = mf.BlochHamiltonian(disk_hamiltonian)
    result = mf.density_matrix(
        h,
        filling=filling,
        keys=[(0, 0)],
        tol=5e-4,
        integration=mf.FermiSimplex(initial_nk=81, max_points=100_000),
    )
    mu = np.sqrt(mass**2 + filling - 1)
    polarization = 2 * mass * (np.sqrt(mass**2 + 1) - mu)
    expected = np.diag([(filling - polarization) / 2, (filling + polarization) / 2])
    error = np.max(np.abs(result.to_tb()[(0, 0)] - expected))
    # The analytic normalized disk average establishes absolute accuracy.
    # Density and mu use separate integration/root approximations.
    assert error < 2e-4, f"analytic density error: {error}"
    assert result.mu == pytest.approx(mu, abs=5e-4)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_callable_dimensions_and_energy_normalization(dimension):
    functions = (
        lambda kx: np.diag([-2.0, 1.0]),
        lambda kx, ky: np.diag([-2.0, 1.0]),
        lambda kx, ky, kz: np.diag([-2.0, 1.0]),
    )
    h = mf.BlochHamiltonian(functions[dimension - 1])
    assert (h.ndim, h.ndof) == (dimension, 2)
    key = (0,) * dimension
    interaction = mf.BilinearInteraction([mf.BilinearTerm(0, np.eye(2), np.eye(2))])
    model = mf.Model(h, interaction, 1)
    result = mf.density_matrix(model, keys=[key], compute_free_energy=True, tol=1e-8)
    np.testing.assert_allclose(result.to_tb()[key], np.diag([1, 0]), atol=1e-14)
    assert result.internal_energy == pytest.approx(-1, abs=1e-14)
    assert result.free_energy == result.internal_energy
    assert (
        mf.density_matrix(h, filling=1, interaction=interaction).coordinates.size == 2
    )


def test_callable_validation():
    for function in (
        lambda: np.eye(2),
        lambda *k: np.eye(2),
        lambda kx, ky=0: np.eye(2),
        lambda kx, *, mass: np.eye(2),
        lambda **k: np.eye(2),
    ):
        with pytest.raises(TypeError, match="required positional"):
            mf.BlochHamiltonian(function)
    with pytest.raises(TypeError, match="callable"):
        mf.BlochHamiltonian(np.eye(2))
    h = mf.BlochHamiltonian(lambda kx: np.eye(2))
    for coordinates in ((), (1, 2), ([1, 2],)):
        with pytest.raises(ValueError, match="coordinates"):
            h(*coordinates)
    for value, message in (
        (np.ones((2, 3)), "square"),
        (np.empty((0, 0)), "nonempty"),
    ):
        with pytest.raises(ValueError, match=message):
            mf.BlochHamiltonian(lambda kx: value)
        # The same validation applies beyond the origin.
        bad = mf.BlochHamiltonian(lambda kx: np.eye(2) if kx == 0 else value)
        with pytest.raises(ValueError, match=message):
            bad(1.0)
    changing_size = mf.BlochHamiltonian(lambda kx: np.eye(2 if kx == 0 else 3))
    with pytest.raises(ValueError, match="shape must remain constant"):
        changing_size(1.0)
    with pytest.raises(FrozenInstanceError):
        h.ndof = 3
    with pytest.raises(ValueError, match="dimension"):
        mf.density_matrix(h, filling=1, keys=[(0, 0)])
    with pytest.raises(ValueError, match="size"):
        mf.density_matrix(
            h,
            filling=1,
            interaction=mf.BilinearInteraction(
                [mf.BilinearTerm(1, np.eye(3), np.eye(3))]
            ),
        )


def test_callable_infers_bound_parameters_and_callable_objects():
    def dispersion(mass, kx, ky, /):
        return np.diag([mass + np.cos(kx), -mass + np.sin(ky)])

    h = mf.BlochHamiltonian(partial(dispersion, 0.2))
    assert (h.ndim, h.ndof) == (2, 2)
    np.testing.assert_allclose(h(0.3, 0.4), dispersion(0.2, 0.3, 0.4), atol=0)

    class Dispersion:
        def __call__(self, kx):
            return np.diag([np.cos(kx), 2.0, 3.0])

    h = mf.BlochHamiltonian(Dispersion())
    assert (h.ndim, h.ndof) == (1, 3)
    result = mf.density_matrix(h, filling=1, keys=[(0,)], tol=1e-6)
    np.testing.assert_allclose(result.to_tb()[(0,)], np.diag([1, 0, 0]), atol=1e-13)
