"""Same-mesh energy rules against exact integrals and independent references."""

from types import SimpleNamespace

import numpy as np
import pytest
from fermisimplex import SpectralMesh

import meanfi as mf
from meanfi.density.integrate.simplex.energy import integrate_energies
from meanfi.meanfield import correction_expectation, interaction_energy


pytestmark = pytest.mark.physics
LOCAL = (0, 0)


def coordinate_function(function, dimension):
    """Give a vector callback the explicit signature required by Hamiltonians."""
    return {
        1: lambda x: function(np.array([x])),
        2: lambda x, y: function(np.array([x, y])),
        3: lambda x, y, z: function(np.array([x, y, z])),
        4: lambda x, y, z, w: function(np.array([x, y, z, w])),
    }[dimension]


def prepared_mesh(function, level=0):
    mesh = SpectralMesh(function, root_level=level)
    mesh.integrate_density_matrix(
        mu=0, lattice_vectors=[(0,) * mesh.ndim], target_error=1e-8, preview_depth=0
    )
    return mesh


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_q2_integrates_all_quadratic_monomials_exactly(dimension):
    powers = [np.zeros(dimension, dtype=int), *np.eye(dimension, dtype=int)]
    powers.extend(
        np.eye(dimension, dtype=int)[i] + np.eye(dimension, dtype=int)[j]
        for i in range(dimension)
        for j in range(i, dimension)
    )
    for power in powers:
        mesh = prepared_mesh(
            coordinate_function(
                lambda x: np.diag([-1 - np.prod(x**power), 2.0]), dimension
            )
        )
        points, simplices = mesh.points.copy(), mesh.simplices.copy()
        cached = mesh.cached_vertices
        result = integrate_energies(mesh, mu=0.3, filling=1)
        exact = (-1 - 1 / np.prod(power + 1)) / 2
        assert result.band_energy == pytest.approx(exact, abs=2e-14)
        assert mesh.cached_vertices == cached
        np.testing.assert_array_equal(mesh.points, points)
        np.testing.assert_array_equal(mesh.simplices, simplices)


def test_centers_are_deduplicated_and_vertex_spectra_reused(monkeypatch):
    calls, batches = [], []

    def evaluate(x):
        calls.append(x)
        return np.diag([-1 - x * x, 1.0])

    mesh = SimpleNamespace(
        ndim=1,
        ndof=2,
        dyadic_numerators=np.array([[0], [1], [1], [1]]),
        dyadic_levels=np.array([0, 0, 1, 2]),
        volumes=np.array([1.0, 0.5, 0.5, 0.5]),
        points=np.array([[0.0], [1.0], [0.5], [0.25]]),
        simplices=np.array([[0, 1], [0, 2], [2, 1], [2, 1]]),
        eigenvalues=np.array([[-1.0, 1.0], [-2.0, 1.0], [-1.25, 1.0], [-1.0625, 1.0]]),
        evaluate=evaluate,
    )
    mesh.evaluated_snapshot = lambda *, include_eigenvectors: mesh
    original = np.linalg.eigvalsh

    def eigenvalues(matrices):
        batches.append(matrices.shape)
        return original(matrices)

    monkeypatch.setattr(np.linalg, "eigvalsh", eigenvalues)
    result = integrate_energies(mesh, mu=0, filling=1)
    assert calls == [0.75]
    assert batches == [(1, 2, 2)]
    assert result.evaluations == 1


def test_selected_density_trace_uses_occupied_weights_without_refining():
    from meanfi.density.integrate.simplex.mesh import SimplexEvaluator
    from meanfi.density.problem import build_density_problem
    from meanfi.space.coordinates import DensityCoordinates

    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,)],
        pairs_by_key={(0,): (np.array([0]), np.array([1]))},
    )
    problem = build_density_problem(
        {
            (0,): np.array([[0.2, 0.1], [0.1, -0.2]]),
            (1,): np.eye(2) * 0.2,
            (-1,): np.eye(2) * 0.2,
        },
        kT=0,
        integration=mf.FermiSimplex(),
        tolerances=mf.default_solver_tolerances(3e-3),
        density_coordinates=coordinates,
    )
    evaluator = SimplexEvaluator(problem)
    density = evaluator.density(0.1)
    before = evaluator.work.diagonalizations, evaluator.mesh.active_simplices
    trace = evaluator.density_trace(0.1, density)
    assert before == (evaluator.work.diagonalizations, evaluator.mesh.active_simplices)
    assert trace == pytest.approx(
        float(np.sum(evaluator.mesh.occupied_weights(0.1))), abs=1e-14
    )


@pytest.mark.parametrize("dimension", [0, 1, 2, 3, 4])
def test_model_density_and_ediis_energy_match_exact_constant_projector(dimension):
    local = (0,) * dimension
    model = mf.Model(
        {local: np.diag([-1.0, 1.0])},
        {local: np.array([[0.0, 0.3], [0.3, 0.0]])},
        filling=1,
    )
    density = mf.density_matrix_at_mu(model, 0, keys=[local], tol=1e-6)
    np.testing.assert_allclose(density.to_tb()[local], np.diag([1.0, 0.0]), atol=1e-13)
    assert density.internal_energy == pytest.approx(-0.5, abs=1e-13)
    result = mf.solver(
        model,
        {local: np.zeros((2, 2))},
        integration=mf.FermiSimplex(),
        scf=mf.EnergyDIIS(),
        tol=1e-6,
    )
    assert result.converged
    assert result.internal_energy == pytest.approx(-0.5, abs=1e-13)
    if dimension:
        assert result.density.statistics.n_energy_evaluations > 0
        assert result.density.statistics.requested_nk is None


@pytest.mark.parametrize("kind", ["normal", "reference", "intercell"])
def test_one_body_energy_does_not_change_scf_density_or_interaction(monkeypatch, kind):
    import meanfi.density.integrate.simplex as simplex

    h0 = mf.BlochHamiltonian(
        lambda kx, ky: np.array(
            [[0.3, 0.12 * np.exp(1j * kx)], [0.12 * np.exp(-1j * kx), -0.3]]
        )
    )
    interaction = {LOCAL: np.array([[0.0, 0.15], [0.15, 0.0]])}
    correction = {LOCAL: np.diag([0.2, -0.1])}
    reference = None
    if kind == "reference":
        from meanfi.tests.fixtures.models import density_result_from_tb

        reference = density_result_from_tb(
            {LOCAL: np.array([[0.3, 0.01j], [-0.01j, 0.6]])}
        )
    if kind in ("intercell", "external"):
        for key in [(1, 0), (-1, 0)]:
            if kind == "intercell":
                interaction[key] = np.full((2, 2), 0.03)
            correction[key] = np.diag([0.01, -0.01])
    model = mf.Model(h0, interaction, filling=1, reference=reference)
    result = mf.density_matrix(model, mean_field=correction, tol=3e-3)
    assert result.band_energy is not None
    state = model._density_state(result)
    exact_interaction = interaction_energy(
        model._active_density_from_state(model._reference_difference(state)),
        model.h_int,
        electron_ndof=model._electron_ndof,
    )
    assert result.internal_energy == pytest.approx(
        result.band_energy
        - correction_expectation(model._active_density_from_state(state), correction)
        + exact_interaction
    )
    assert (
        result.select(model.required_coordinates).internal_energy
        == result.internal_energy
    )
    monkeypatch.setattr(
        simplex,
        "integrate_energies",
        lambda mesh, **kwargs: SimpleNamespace(
            band_energy=0.0, evaluations=0, simplices=0
        ),
    )
    baseline = mf.density_matrix(model, mean_field=correction, tol=3e-3)
    np.testing.assert_array_equal(result.values, baseline.values)
    assert result.errors == baseline.errors
    assert (
        result.statistics.n_diagonalizations
        == baseline.statistics.n_diagonalizations
        + result.statistics.n_energy_evaluations
    )
    assert result.statistics.refinements == baseline.statistics.refinements
    assert result.statistics.n_energy_evaluations > 0


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_ediis_history_uses_relative_energy_and_original_density(
    monkeypatch, dimension
):
    import meanfi.scf.engine as engine
    from meanfi.scf.problem import SCFProblem
    from meanfi.scf.energy import energy_sample, comparison_points

    local = (0,) * dimension
    h0 = mf.BlochHamiltonian(
        coordinate_function(
            lambda k: np.array(
                [[0.04, 0.1 * np.exp(1j * k[0])], [0.1 * np.exp(-1j * k[0]), -0.04]]
            ),
            dimension,
        )
    )
    model = mf.Model(h0, {local: np.array([[0.0, 0.12], [0.12, 0.0]])}, filling=1)
    samples = []
    evaluate = SCFProblem.evaluate_state
    original = engine.ediis_coefficients

    def record(self, state, mu_guess):
        result = evaluate(self, state, mu_guess)
        assert result.internal_energy is None
        assert result.density.band_energy is None
        samples.append(energy_sample(model, result))
        return result

    def coefficients(points, **kwargs):
        expected = comparison_points(model, samples[-len(points) :])
        for point, reference in zip(points, expected, strict=True):
            np.testing.assert_array_equal(point.params, reference.params)
            assert point.energy == reference.energy
        return original(points, **kwargs)

    monkeypatch.setattr(SCFProblem, "evaluate_state", record)
    monkeypatch.setattr(engine, "ediis_coefficients", coefficients)
    result = mf.solver(
        model, {local: np.diag([0.02, -0.02])}, scf=mf.EnergyDIIS(), tol=3e-3
    )
    assert result.converged and len(samples) > 1
    assert np.isfinite(result.internal_energy)


def test_fivefold_energy_against_converged_independent_quadrature():
    from scipy.special import roots_legendre
    from meanfi.meanfield import correction_expectation

    bare_mass, effective_mass = 0.002, 0.02

    def bare(kx, ky):
        z = (kx / np.pi - 1 + 1j * (ky / np.pi - 1)) ** 5
        return np.array([[bare_mass, z.conjugate()], [z, -bare_mass]])

    h0 = mf.BlochHamiltonian(bare)
    model = mf.Model(h0, {LOCAL: np.array([[0.0, 0.15], [0.15, 0.0]])}, filling=1)
    sigma = {LOCAL: np.diag([effective_mass - bare_mass, bare_mass - effective_mass])}
    result = mf.density_matrix_at_mu(model, 0, mean_field=sigma, tol=3e-3)
    references = []
    total_references = []
    for order in (128, 256):
        x, weights = roots_legendre(order)
        radius10 = (x[:, None] ** 2 + x[None, :] ** 2) ** 5
        bare_energy = -(radius10 + bare_mass * effective_mass) / np.sqrt(
            radius10 + effective_mass**2
        )
        references.append(float(weights @ bare_energy @ weights) / 8)
        polarization = (
            float(
                weights
                @ (effective_mass / np.sqrt(radius10 + effective_mass**2))
                @ weights
            )
            / 4
        )
        reference_density = np.diag([(1 - polarization) / 2, (1 + polarization) / 2])
        total_references.append(
            references[-1] + interaction_energy({LOCAL: reference_density}, model.h_int)
        )
    assert abs(references[1] - references[0]) < 1e-11
    assert abs(total_references[1] - total_references[0]) < 1e-11
    state = model._active_density_from_state(model._density_state(result))
    improved = result.band_energy - correction_expectation(state, sigma)
    improved_error = abs(improved - references[-1])
    print(f"Fivefold Q2 one-body error per orbital: {improved_error:.6g}")
    # This loose density tolerance targets qualitative energies: <0.002 per orbital.
    assert improved_error < 2e-3
    total_error = abs(result.internal_energy - total_references[-1])
    print(f"Fivefold Q2 total HF error per orbital: {total_error:.6g}")
    assert total_error < 2e-3


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_energy_uses_complete_preview_partition_without_exporting_vectors(
    monkeypatch, dimension
):
    from fractions import Fraction

    mesh = SpectralMesh(
        coordinate_function(lambda x: np.diag([-1 - x[0] ** 2, 2.0]), dimension),
        root_level=0,
    )
    mesh.integrate_density_matrix(
        mu=0, lattice_vectors=[(0,) * dimension], target_error=1e-8, preview_depth=1
    )
    snapshot = mesh.evaluated_snapshot(include_eigenvectors=False)
    assert snapshot.preview_vertices > 0
    assert snapshot.partition_simplices > mesh.active_simplices
    keys = [
        tuple(Fraction(int(n), 1 << int(level)) for n in row)
        for row, level in zip(snapshot.dyadic_numerators, snapshot.dyadic_levels)
    ]
    centers = {
        tuple(
            sum(keys[i][axis] for i in simplex) / len(simplex)
            for axis in range(dimension)
        )
        for simplex in snapshot.simplices
    }
    missing = centers - set(keys)
    exported = []
    original = mesh.evaluated_snapshot

    def export(*, include_eigenvectors):
        exported.append(include_eigenvectors)
        return original(include_eigenvectors=include_eigenvectors)

    monkeypatch.setattr(mesh, "evaluated_snapshot", export)
    cached = mesh.cached_vertices
    energy = integrate_energies(mesh, mu=0, filling=1)
    assert energy.band_energy == pytest.approx(-2 / 3, abs=2e-14)
    assert energy.evaluations == len(missing)
    assert energy.simplices == snapshot.partition_simplices
    assert exported == [False]
    assert mesh.cached_vertices == cached


def test_preview_q2_improves_quartic_integral_on_unchanged_active_mesh():
    mesh = SpectralMesh(lambda x: np.diag([-1 - x**4, 2.0]), root_level=0)
    mesh.integrate_density_matrix(
        mu=0, lattice_vectors=[(0,)], target_error=1e-8, preview_depth=0
    )
    coarse = integrate_energies(mesh, mu=0, filling=1)
    active = mesh.simplices.copy()
    mesh.integrate_density_matrix(
        mu=0, lattice_vectors=[(0,)], target_error=1e-8, preview_depth=1
    )
    fine = integrate_energies(mesh, mu=0, filling=1)
    np.testing.assert_array_equal(mesh.simplices, active)
    exact = -0.6
    assert abs(fine.band_energy - exact) < abs(coarse.band_energy - exact) / 10
