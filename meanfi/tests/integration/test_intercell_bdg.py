"""Normal/intercell/BdG workflows through the shared solver and UniformGrid."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq, root

import meanfi as mf
from meanfi.scf.problem import SCFProblem
from meanfi.tb.bdg import validate_bdg_tb

pytestmark = pytest.mark.integration


def _chain_model(superconducting, kT):
    hop = np.array([[-0.7, 0.06j], [0.08, -0.5]])
    h0 = {
        (0,): np.array([[0.3, 0.09j], [-0.09j, -0.2]]),
        (1,): hop,
        (-1,): hop.conj().T,
    }
    a = np.array([[0.2, 0.3j], [-0.3j, 0.7]])
    b = np.array([[0.6, 0.1], [0.1, -0.1]])
    interaction = mf.BilinearInteraction(
        [
            mf.BilinearTerm(-0.2, a, b),
            mf.BilinearTerm(0.1, a, b, displacement=(1,)),
        ]
    )
    return mf.Model(h0, interaction, 0.8, kT=kT, superconducting=superconducting)


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("kT", [0.0, 0.2])
def test_callable_intercell_density_and_energy_match_tb(superconducting, kT):
    tb = _chain_model(superconducting, kT)
    function = mf.tb_to_kfunc(tb.h_0)
    model = replace(tb, h_0=mf.BlochHamiltonian(lambda kx: function([kx])))
    correction = model.random_meanfield(rng=83, scale=0.1)
    integration = mf.UniformGrid(nk=64)
    kwargs = dict(
        mu=0.13,
        keys=[(0,), (1,), (-1,)],
        mean_field=correction,
        integration=integration,
        compute_free_energy=True,
        tol=1e-8,
    )
    expected = mf.density_matrix_at_mu(tb, **kwargs)
    actual = mf.density_matrix_at_mu(model, **kwargs)
    np.testing.assert_allclose(actual.values, expected.values, atol=3e-13, rtol=0)
    assert actual.internal_energy == pytest.approx(
        mf.evaluate_internal_energy(tb, actual), abs=3e-13
    )
    assert actual.entropy == pytest.approx(expected.entropy, abs=3e-13)
    if superconducting:
        hk = model.hamiltonian_from_meanfield(correction)
        swap = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
        np.testing.assert_allclose(
            hk(0.7), -swap @ hk(2 * np.pi - 0.7).conj() @ swap, atol=3e-14
        )
        validate_bdg_tb(correction, ndof=2, ndim=1)
        with pytest.raises(
            ValueError, match="Superconducting density requires UniformGrid"
        ):
            mf.density_matrix_at_mu(model, 0, integration=mf.FermiSimplex())
    elif kT == 0:
        # The new normal interaction also selects the right Fourier moments in FermiSimplex.
        kwargs.update(integration=mf.FermiSimplex(nk=129))
        expected = mf.density_matrix_at_mu(tb, **kwargs)
        actual = mf.density_matrix_at_mu(model, **kwargs)
        np.testing.assert_allclose(actual.values, expected.values, atol=3e-13, rtol=0)


@pytest.mark.parametrize("paired_reference", [False, True])
def test_callable_bdg_reference_energy_and_ediis_curvature(paired_reference):
    tb = _chain_model(True, 0.2)
    integration = mf.UniformGrid(nk=32)
    full = mf.density_matrix_at_mu(
        tb,
        0.1,
        keys=[(0,), (1,), (-1,)],
        mean_field=tb.random_meanfield(rng=6, scale=0.1),
        integration=integration,
    )
    reference = (
        full
        if paired_reference
        else {key: block[:2, :2] for key, block in full.to_tb().items()}
    )
    tb = replace(tb, reference=reference)
    function = mf.tb_to_kfunc(tb.h_0)
    model = replace(tb, h_0=mf.BlochHamiltonian(lambda kx: function([kx])))
    correction = model.random_meanfield(rng=4, scale=0.2)
    tolerance = replace(
        mf.default_solver_tolerances(1e-7), filling_residual=1e-12, mu_tol=1e-12
    )
    actual = mf.density_matrix(
        model,
        keys=[(0,), (1,), (-1,)],
        mean_field=correction,
        integration=integration,
        tol=tolerance,
        compute_free_energy=True,
    )
    expected = mf.density_matrix(
        tb,
        keys=[(0,), (1,), (-1,)],
        mean_field=correction,
        integration=integration,
        tol=tolerance,
        compute_free_energy=True,
    )
    np.testing.assert_allclose(actual.values, expected.values, atol=3e-12, rtol=0)
    assert actual.internal_energy == pytest.approx(
        mf.evaluate_internal_energy(tb, actual), abs=3e-13
    )
    assert actual.free_energy == pytest.approx(
        mf.evaluate_free_energy(tb, actual), abs=3e-13
    )
    rho, sigma = actual.to_tb(), full.to_tb()
    weight = 0.3
    mixed = {key: weight * rho[key] + (1 - weight) * sigma[key] for key in rho}
    direction = model._density_state(actual).values - model._density_state(full).values
    curvature = SCFProblem(model, density_problem=None).interaction_curvature(direction)
    energy = (
        weight * mf.evaluate_internal_energy(tb, actual)
        + (1 - weight) * mf.evaluate_internal_energy(tb, full)
        - weight * (1 - weight) * curvature
    )
    assert energy == pytest.approx(mf.evaluate_internal_energy(tb, mixed), abs=3e-13)


def test_local_callable_bdg_scf_matches_analytic_gap():
    temperature, coupling = 0.1, -1.0
    h = mf.BlochHamiltonian(lambda kx: np.zeros((2, 2)))
    interaction = mf.BilinearInteraction(
        [mf.BilinearTerm(coupling, np.diag([1, 0]), np.diag([0, 1]))]
    )
    model = mf.Model(h, interaction, 1, kT=temperature, superconducting=True)
    pairing = np.array([[0, 0.2j], [-0.2j, 0]])
    guess = {
        (0,): np.block(
            [[np.zeros((2, 2)), pairing], [pairing.conj().T, np.zeros((2, 2))]]
        )
    }
    result = mf.solver(model, guess, tol=1e-8, compute_free_energy=True)
    exact_gap = brentq(
        lambda gap: gap - abs(coupling) / 2 * np.tanh(gap / (2 * temperature)),
        1e-5,
        0.5,
        xtol=1e-14,
    )
    error = abs(abs(result.mean_field[(0,)][0, 3]) - exact_gap)
    assert result.converged
    assert error < 2e-8, f"Analytic superconducting gap error: {error}"
    assert result.mu == pytest.approx(coupling / 2, abs=2e-9)
    exact_energy = coupling * (0.25 + (exact_gap / coupling) ** 2) / 2
    assert result.internal_energy == pytest.approx(exact_energy, abs=2e-9)
    assert result.free_energy == pytest.approx(
        result.internal_energy - temperature * result.entropy, abs=1e-14
    )


def test_intercell_callable_bdg_scf_matches_density_density_chain():
    coupling = -1.5
    h = mf.BlochHamiltonian(lambda kx: np.array([[-2 * np.cos(kx)]]))
    interaction = mf.BilinearInteraction(
        [mf.BilinearTerm(coupling, np.eye(1), np.eye(1), displacement=(1,))]
    )
    model = mf.Model(h, interaction, 0.5, kT=0.08, superconducting=True)
    legacy = replace(
        model,
        h_0={(0,): np.zeros((1, 1)), (1,): -np.eye(1), (-1,): -np.eye(1)},
        h_int={(1,): coupling * np.eye(1), (-1,): coupling * np.eye(1)},
    )
    guess = {
        (1,): np.array([[0, 0.2], [-0.2, 0]]),
        (-1,): np.array([[0, -0.2], [0.2, 0]]),
    }
    integration = mf.UniformGrid(nk=128)
    tolerance = replace(
        mf.default_solver_tolerances(1e-7), filling_residual=1e-12, mu_tol=1e-12
    )
    result = mf.solver(model, guess, integration=integration, tol=tolerance)
    expected = mf.solver(legacy, guess, integration=integration, tol=tolerance)
    assert result.converged
    assert abs(result.mean_field[(1,)][0, 1]) > 0.05
    for key, value in result.mean_field.items():
        np.testing.assert_allclose(value, expected.mean_field[key], atol=2e-8, rtol=0)
    assert result.internal_energy == pytest.approx(expected.internal_energy, abs=2e-9)

    # Independent spinless gap equations at half filling. Establish the reference
    # by doubling its dense grid; no MeanFi contractions or eigensolver are used.
    def reference(nk):
        k = 2 * np.pi * np.arange(nk) / nk

        def equations(values):
            bond_density, gap = values
            xi = -2 * (1 + coupling * bond_density) * np.cos(k)
            energy = np.sqrt(xi**2 + 4 * gap**2 * np.sin(k) ** 2)
            weight = np.tanh(energy / (2 * model.kT)) / energy
            return [
                bond_density - np.mean(np.cos(k) * (1 - xi * weight) / 2),
                1 + coupling * np.mean(np.sin(k) ** 2 * weight),
            ]

        solution = root(equations, [0.28, 0.3], tol=1e-11)
        assert solution.success
        return solution.x

    coarse, fine = reference(512), reference(1024)
    np.testing.assert_allclose(coarse, fine, atol=2e-13, rtol=0)
    bond_density, gap = fine
    gap_error = abs(result.mean_field[(1,)][0, 1] - gap)
    assert gap_error < 2e-7, f"Dense-reference intercell gap error: {gap_error}"
    reference_energy = -2 * bond_density + coupling * (
        0.25 - bond_density**2 + (gap / coupling) ** 2
    )
    assert result.internal_energy == pytest.approx(reference_energy, abs=2e-9)


def test_callable_bdg_charge_bracket_expands_beyond_origin_scale():
    h = mf.BlochHamiltonian(lambda kx: np.array([[50 * (1 - np.cos(kx))]]))
    interaction = mf.BilinearInteraction([mf.BilinearTerm(0, np.eye(1), np.eye(1))])
    model = mf.Model(h, interaction, 0.8, kT=0.2, superconducting=True)
    integration = mf.UniformGrid(nk=64)
    result = mf.density_matrix(model, integration=integration, tol=1e-8)
    expected = mf.density_matrix(
        h, filling=0.8, kT=0.2, keys=[(0,)], integration=integration, tol=1e-8
    )
    assert result.mu > 50
    assert result.mu == pytest.approx(expected.mu, abs=2e-6)
    assert result.filling == pytest.approx(0.8, abs=1e-9)


def test_zero_temperature_bdg_retains_uniform_grid_filling_contract():
    model = _chain_model(True, 0.0)
    with pytest.raises(NotImplementedError, match="UniformGrid fixed-filling"):
        mf.density_matrix(model, integration=mf.UniformGrid(nk=16))
