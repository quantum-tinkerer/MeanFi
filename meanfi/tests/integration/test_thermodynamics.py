"""Thermodynamics against spectra and full-density energy functionals."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import entr, expit

import meanfi as mf
from meanfi.tb.bdg import assemble_bdg_tb

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("superconducting", [False, True])
def test_periodic_thermodynamics_reuses_density_spectrum(superconducting, monkeypatch):
    h0 = {(): np.array([[0.3, 0.04j], [-0.04j, -0.2]])}
    interaction = {(): np.array([[0.0, 1.5], [1.5, 0.0]])}
    model = mf.Model(
        h0, interaction, filling=0.8, kT=0.17, superconducting=superconducting
    )
    correction = {(): np.array([[0.1, 0.02j], [-0.02j, 0.2]])}
    if superconducting:
        correction = assemble_bdg_tb(
            correction, {(): np.array([[0, 0.2j], [-0.2j, 0]])}, ndof=2
        )
    h = model.hamiltonian_from_meanfield(correction)[()]
    mu = 0.12
    q = np.array([1, 1, -1, -1]) if superconducting else np.ones(2)
    energies, vectors = np.linalg.eigh(h - mu * np.diag(q))
    occupations = expit(-energies / model.kT)
    full = (vectors * occupations) @ vectors.conj().T
    factor = 0.5 if superconducting else 1.0
    expected_band = factor * np.trace(h @ full).real
    if superconducting:
        expected_band += 0.5 * np.trace(h[:2, :2]).real
    expected_entropy = factor * np.sum(entr(occupations) + entr(1 - occupations))
    calls = []
    eigh = np.linalg.eigh

    def counted(*args, **kwargs):
        calls.append(1)
        return eigh(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    density = mf.density_matrix_at_mu(model, mu=mu, mean_field=correction)
    assert len(calls) == 1
    assert not density.is_complete
    assert density.band_energy == pytest.approx(expected_band / 2, abs=1e-13)
    assert density.entropy == pytest.approx(expected_entropy / 2, abs=1e-13)


@pytest.mark.parametrize("dimension", [0, 1, 2, 3])
def test_zero_temperature_flat_band_retains_half_occupation_entropy(dimension):
    key = (0,) * dimension
    result = mf.density_matrix_at_mu(
        {key: np.diag([0.0, 1.0])}, mu=0.0, kT=0.0, keys=[key]
    )
    assert result.filling == pytest.approx(0.5)
    assert result.entropy == pytest.approx(np.log(2.0) / 2)
    assert result.band_energy == pytest.approx(0.0)


@pytest.mark.parametrize("superconducting", [False, True])
def test_scf_selected_energy_agrees_with_full_density(superconducting):
    h0 = {(0,): np.diag([0.2, -0.3]), (1,): np.array([[-0.3, 0.05j], [0.0, -0.2]])}
    h0[(-1,)] = h0[(1,)].conj().T
    hint = {(0,): np.array([[0.0, 0.5], [0.5, 0.0]])}
    model = mf.Model(h0, hint, filling=0.8, kT=0.2, superconducting=superconducting)
    grid = mf.PeriodicGrid(nk=32)
    guess = model.random_meanfield(rng=np.random.default_rng(4), scale=0.05)
    if superconducting:
        # A user may supply zero corrections on the full hopping support.
        guess.update({key: np.zeros((4, 4)) for key in h0 if key not in guess})
    result = mf.solver(model, guess, integration=grid, tol=1e-8)
    full = mf.density_matrix(
        model, mean_field=result.mean_field, keys=list(h0), integration=grid, tol=1e-10
    )
    assert result.converged
    assert result.internal_energy == pytest.approx(
        mf.internal_energy(model, full), abs=1e-8
    )
    assert result.entropy == pytest.approx(full.entropy, abs=1e-8)
    assert result.free_energy == pytest.approx(mf.free_energy(model, full), abs=1e-8)
    assert all(
        point.free_energy
        == pytest.approx(point.internal_energy - model.kT * point.entropy)
        for point in result.history
    )


@pytest.mark.parametrize("a,b,kT", [(0.2, 3.0, 0.2), (0.5, 5.0, 0.1)])
def test_finite_temperature_ediis_finishes_when_entropy_history_stalls(a, b, kT):
    model = mf.Model(
        {(): np.diag([a, -a])},
        {(): np.array([[0.0, -2 * b], [-2 * b, 0.0]])},
        filling=1.0,
        kT=kT,
    )
    expected_q = brentq(lambda q: q + np.tanh((a + b * q) / (2 * kT)), -1, 1)
    result = mf.solver(
        model,
        {(): np.diag([0.5, -0.5])},
        tol=1e-9,
        scf=mf.EnergyDIIS(max_iterations=60),
    )
    values = (
        result.density.to_tb()[()]
        if result.density.is_complete
        else model._active_density_from_state(model._density_state(result.density))[()]
    )
    assert (values[0, 0] - values[1, 1]).real == pytest.approx(expected_q, abs=1e-8)
    assert result.errors.scf_residual <= 1e-9
    assert len(result.history) <= 60
    assert result.free_energy < result.internal_energy


def test_reference_subtraction_does_not_subtract_entropy():
    h0 = {(): np.diag([-0.2, 0.2])}
    hint = {(): np.array([[0.0, 0.5], [0.5, 0.0]])}
    bare = mf.Model(h0, hint, filling=1.0, kT=0.2)
    reference = mf.density_matrix(bare, keys=[()])
    model = replace(bare, reference=reference)
    result = mf.solver(model, {(): np.zeros((2, 2))}, tol=1e-9)
    assert result.entropy == pytest.approx(reference.entropy)
    assert result.internal_energy == pytest.approx(
        mf.expectation_value(reference, h0).real / 2
    )
    assert result.free_energy == pytest.approx(
        result.internal_energy - model.kT * reference.entropy
    )


def test_empty_density_selection_reports_thermal_errors_without_refining_them():
    h = {(0,): np.zeros((1, 1)), (1,): np.array([[0.5]]), (-1,): np.array([[0.5]])}
    coordinates = mf.DensityCoordinates.from_entries(size=1, keys=[(0,)], entries=())
    result = mf.density_matrix_at_mu(
        h, mu=0.0, kT=0.1, coordinates=coordinates, tol=1e-5
    )
    reference = mf.density_matrix_at_mu(
        h, mu=0.0, kT=0.1, coordinates=coordinates, integration=mf.PeriodicGrid(nk=8192)
    )
    assert result.values.size == 0
    assert result.errors.density_matrix_integration == 0.0
    assert result.errors.charge_integration <= 2e-6
    # No density entries were requested. Charge is constant at half filling,
    # so thermal discrepancies must not force more integration work.
    assert abs(result.band_energy - reference.band_energy) > 1e-3
    assert abs(result.entropy - reference.entropy) > 1e-3
    assert result.errors.band_energy_integration > 1e-3
    assert result.errors.entropy_integration > 1e-3


@pytest.mark.parametrize("use_sparse", [False, True])
def test_bdg_shifted_grid_energy_uses_full_nambu_charge(use_sparse, request):
    from scipy import sparse
    from meanfi.density.integrate.periodic_grid import _Evaluator, _Grid
    from meanfi.errors import default_solver_tolerances

    h0 = {
        (0,): np.array([[0.2]]),
        (1,): np.array([[-0.3 + 0.2j]]),
        (-1,): np.array([[-0.3 - 0.2j]]),
    }
    pairing = {(1,): np.array([[0.1]]), (-1,): np.array([[-0.1]])}
    h = assemble_bdg_tb(h0, pairing, ndof=1)
    coordinates = mf.DensityCoordinates.from_entries(size=2, keys=[(0,)], entries=())
    grid = _Grid(3, 1, shifted=True)
    points = next(grid.batches(3))[2]
    matrices = mf.tb_to_kfunc(h)(points)
    q = np.array([1.0, -1.0])
    energies, vectors = np.linalg.eigh(matrices - 0.15 * np.diag(q))
    occupations = expit(-energies / 0.12)
    full = (vectors * occupations[:, None, :]) @ vectors.conj().swapaxes(-1, -2)
    reference = np.mean(
        0.5
        * (np.trace(matrices @ full, axis1=-2, axis2=-1).real + matrices[:, 0, 0].real)
    )
    if use_sparse:
        request.getfixturevalue("require_mumps")
        h = {key: sparse.csr_matrix(block) for key, block in h.items()}
    evaluator = _Evaluator(
        h,
        kT=0.12,
        integration=mf.PeriodicGrid(
            nk=3,
            matrix_function=mf.RationalFOE()
            if use_sparse
            else mf.DirectDiagonalization(),
        ),
        coordinates=coordinates,
        q_diag=q,
        trace_weights=np.array([1.0, 0.0]),
        tolerances=default_solver_tolerances(1e-8),
    )
    result, _ = evaluator.density(grid, 0.15, compare_previous=False)
    assert result.band_energy == pytest.approx(reference, abs=1e-8)


def test_bdg_interaction_support_does_not_depend_on_guess_keys():
    h0 = {(0,): np.zeros((1, 1)), (1,): np.array([[-0.4]]), (-1,): np.array([[-0.4]])}
    hint = {(1,): np.array([[0.3]]), (-1,): np.array([[0.3]])}
    model = mf.Model(h0, hint, filling=0.4, kT=0.2, superconducting=True)
    zero = np.zeros((2, 2))
    grid = mf.PeriodicGrid(nk=32)
    minimal = mf.solver(model, {(0,): zero}, integration=grid, tol=1e-8)
    complete = mf.solver(model, {key: zero for key in h0}, integration=grid, tol=1e-8)
    assert set(minimal.mean_field) == set(h0)
    assert abs(minimal.mean_field[(1,)][0, 0]) > 1e-3
    for key in minimal.mean_field:
        np.testing.assert_allclose(
            minimal.mean_field[key], complete.mean_field[key], atol=1e-12
        )
    assert minimal.free_energy == pytest.approx(complete.free_energy)


@pytest.mark.parametrize(
    "superconducting,kT", [(False, 0.0), (False, 0.17), (True, 0.17)]
)
@pytest.mark.parametrize("dimension", [0, 1])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_thermodynamics_per_orbital_is_invariant_under_independent_copies(
    superconducting, kT, dimension, use_sparse, request
):
    from scipy import sparse

    if use_sparse:
        if kT == 0:
            pytest.skip("Sparse density evaluation requires positive temperature")
        request.getfixturevalue("require_mumps")
    key = (0,) * dimension
    h0 = {key: np.array([[0.3, 0.04j], [-0.04j, -0.2]])}
    hint = {key: np.array([[0.0, 0.5], [0.5, 0.0]])}
    normal = {key: np.array([[0.1, 0.02j], [-0.02j, 0.2]])}
    pairing = {key: np.array([[0.0, 0.13j], [-0.13j, 0.0]])}
    if dimension:
        h0[(1,)] = np.diag([-0.3, -0.2])
        h0[(-1,)] = h0[(1,)].conj().T

    def evaluate(copies):
        def repeat(tb):
            blocks = {k: np.kron(np.eye(copies), block) for k, block in tb.items()}
            if use_sparse:
                blocks = {k: sparse.csr_matrix(block) for k, block in blocks.items()}
            return blocks

        model = mf.Model(
            repeat(h0),
            repeat(hint),
            filling=0.8 * copies,
            kT=kT,
            superconducting=superconducting,
        )
        correction = repeat(normal)
        if superconducting:
            correction = assemble_bdg_tb(correction, repeat(pairing), ndof=2 * copies)
        integration = mf.PeriodicGrid(nk=64) if kT > 0 else mf.AdaptiveSimplex()
        density = mf.density_matrix_at_mu(
            model,
            mu=0.12,
            mean_field=correction,
            keys=list(h0),
            integration=integration,
            tol=1e-7,
        )
        selected = density.select(model.scf_space.required_coordinates)
        assert selected.entropy == density.entropy
        return (
            density,
            mf.internal_energy(model, density),
            mf.free_energy(model, density),
        )

    base, base_energy, base_free_energy = evaluate(1)
    repeated, energy, free_energy = evaluate(3)
    assert repeated.filling == pytest.approx(3 * base.filling, abs=1e-7)
    assert repeated.band_energy == pytest.approx(base.band_energy, abs=1e-7)
    assert repeated.entropy == pytest.approx(base.entropy, abs=1e-7)
    assert energy == pytest.approx(base_energy, abs=1e-7)
    assert free_energy == pytest.approx(base_free_energy, abs=1e-7)


@pytest.mark.parametrize("superconducting", [False, True])
def test_ediis_per_orbital_energy_is_invariant_under_independent_copies(
    superconducting,
):
    def solve(copies):
        model = mf.Model(
            {(): np.kron(np.eye(copies), np.diag([0.2, -0.2]))},
            {(): np.kron(np.eye(copies), [[0.0, 0.5], [0.5, 0.0]])},
            filling=float(copies),
            kT=0.2,
            superconducting=superconducting,
        )
        size = 4 * copies if superconducting else 2 * copies
        result = mf.solver(model, {(): np.zeros((size, size))}, tol=1e-9)
        assert result.converged
        return result

    base, repeated = solve(1), solve(3)
    assert repeated.density.filling == pytest.approx(3 * base.density.filling)
    assert repeated.internal_energy == pytest.approx(base.internal_energy, abs=1e-9)
    assert repeated.free_energy == pytest.approx(base.free_energy, abs=1e-9)
    assert repeated.entropy == pytest.approx(base.entropy, abs=1e-9)
