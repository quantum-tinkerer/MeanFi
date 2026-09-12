"""Contract and independent numerical checks for the periodic integration family."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq
from scipy.special import expit

from meanfi import DirectDiagonalization, PeriodicGrid, RationalFOE
from meanfi.density.integrate.periodic import (
    resolve_periodic_matrix_function,
    solve_periodic,
)
from meanfi.space.coordinates import DensityCoordinates


def wire(harmonic=1):
    return {
        (0,): np.array([[0.0]]),
        (harmonic,): np.array([[1.0]]),
        (-harmonic,): np.array([[1.0]]),
    }


def evaluate(hamiltonian=None, *, integration=None, **kwargs):
    return solve_periodic(
        wire() if hamiltonian is None else hamiltonian,
        kT=kwargs.pop("kT", 0.2),
        keys=kwargs.pop("keys", [(0,), (1,)]),
        integration=PeriodicGrid() if integration is None else integration,
        **kwargs,
    )


@pytest.mark.parametrize(
    "dimension,nk,shape",
    [(1, 17, (17,)), (2, 1000, (32, 32)), (2, 4096, (64, 64)), (3, 4096, (16, 16, 16))],
)
def test_nk_is_total_and_fixed_has_no_validation(dimension, nk, shape):
    key = (0,) * dimension
    result = evaluate(
        {key: np.array([[0.3]])}, keys=[key], integration=PeriodicGrid(nk=nk), mu=0.1
    )
    info = result.statistics
    assert info.requested_nk == nk
    assert info.grid_shape == shape
    assert info.n_kpoints == np.prod(shape)
    assert info.n_diagonalizations == np.prod(shape)
    assert info.refinements == info.validation_evaluations == 0
    assert info.density_integration_calls == 1
    assert result.errors.charge_integration is None
    assert result.errors.density_matrix_integration is None
    assert result.density.errors is None


def test_fixed_grid_matches_independent_fourier_sum():
    n, mu, temperature = 29, 0.31, 0.17
    result = evaluate(integration=PeriodicGrid(nk=n), kT=temperature, mu=mu)
    k = 2 * np.pi * np.arange(n) / n
    occupations = expit((mu - 2 * np.cos(k)) / temperature)
    assert_allclose(
        result.density.values,
        [occupations.mean(), (occupations * np.exp(1j * k)).mean()],
        atol=1e-14,
    )


def test_adaptive_fixed_filling_matches_dense_reference():
    filling, temperature = 0.37, 0.19
    k = 2 * np.pi * np.arange(16384) / 16384
    energies = 2 * np.cos(k)
    expected_mu = brentq(
        lambda mu: expit((mu - energies) / temperature).mean() - filling, -3, 3
    )
    expected_density = (
        expit((expected_mu - energies) / temperature) * np.exp(1j * k)
    ).mean()
    result = evaluate(
        kT=temperature,
        filling=filling,
        integration=PeriodicGrid(density_matrix_tol=1e-8, charge_tol=1e-8),
        filling_tol=1e-10,
    )
    assert_allclose(result.mu, expected_mu, atol=2e-9)
    assert_allclose(result.density.values, [filling, expected_density], atol=2e-9)
    assert result.statistics.validation_evaluations > 0
    assert result.errors.filling_residual <= 1e-10
    assert result.errors.charge_integration <= 1e-8


@pytest.mark.parametrize("harmonic", [8, 16])
def test_shifted_grid_rejects_nested_alias(harmonic):
    # Initial 4- and 8-point grids miss both oscillations. A half-cell shift
    # would also miss harmonic 16, so validation needs an irrational offset.
    with pytest.raises(RuntimeError, match="max_refinements=1"):
        evaluate(
            wire(harmonic),
            keys=[(0,)],
            mu=0.3,
            integration=PeriodicGrid(max_refinements=1),
        )
    result = evaluate(wire(harmonic), keys=[(0,)], mu=0.3)
    k = 2 * np.pi * np.arange(32768) / 32768
    reference = expit((0.3 - 2 * np.cos(harmonic * k)) / 0.2).mean()
    assert_allclose(result.filling, reference, atol=2e-6)
    assert result.statistics.n_kpoints > 8


def test_normal_roots_reuse_spectra_across_mu_and_refinement(monkeypatch):
    diagonalized = 0
    original = np.linalg.eigvalsh

    def record(matrices):
        nonlocal diagonalized
        diagonalized += len(matrices)
        return original(matrices)

    monkeypatch.setattr(np.linalg, "eigvalsh", record)
    result = evaluate(filling=0.37)
    assert result.statistics.charge_evaluations > 3
    # Every primary point's eigenvalues were computed only once over all roots.
    assert diagonalized == result.statistics.n_kpoints
    assert result.statistics.n_diagonalizations > diagonalized


def test_batches_bound_transient_eigensystems(monkeypatch):
    sizes = []
    original = np.linalg.eigh

    def record(matrices):
        sizes.append(len(matrices))
        return original(matrices)

    monkeypatch.setattr(np.linalg, "eigh", record)
    evaluate(integration=PeriodicGrid(nk=7, batch_size=3), mu=0.13)
    assert sizes == [3, 3, 1]


def test_full_and_selected_entries_agree_with_direct_reference():
    onsite = np.array([[0.4, 0.3j], [-0.3j, -0.2]])
    hopping = np.array([[0.7, 0.1], [0.2, -0.4]])
    hamiltonian = {(0,): onsite, (1,): hopping, (-1,): hopping.T}
    integration = PeriodicGrid(nk=31)
    coords = DensityCoordinates.from_entries(
        size=2,
        keys=[(0,), (1,)],
        entries=(((0,), 1, 0), ((1,), 0, 1)),
        allow_empty=False,
    )
    full = evaluate(hamiltonian, integration=integration, mu=0.23)
    selected = evaluate(
        hamiltonian, integration=integration, mu=0.23, density_coordinates=coords
    )
    indices = [full.density.coordinates.index(*entry) for entry in coords.entries]
    assert_allclose(selected.density.values, full.density.values[indices], atol=1e-14)
    reference = np.zeros(2, dtype=complex)
    for k in np.arange(31) * 2 * np.pi / 31:
        values, vectors = np.linalg.eigh(
            onsite + hopping * np.exp(-1j * k) + hopping.T * np.exp(1j * k)
        )
        density = (vectors * expit((0.23 - values) / 0.2)) @ vectors.conj().T
        reference += [density[1, 0], density[0, 1] * np.exp(1j * k)]
    assert_allclose(selected.density.values, reference / 31, atol=1e-14)


def test_bdg_recomputes_mu_dependent_spectrum_and_matches_reference(monkeypatch):
    # A paired two-level Nambu Hamiltonian has a mu-dependent eigenbasis.
    hamiltonian = {(0,): np.array([[0.5, 0.3], [0.3, -0.5]])}
    calls = []
    original = np.linalg.eigh

    def record(matrices):
        calls.append(matrices.copy())
        return original(matrices)

    monkeypatch.setattr(np.linalg, "eigh", record)
    result = evaluate(
        hamiltonian,
        keys=[(0,)],
        filling=0.3,
        integration=PeriodicGrid(nk=5),
        q_diag=np.array([1.0, -1.0]),
        trace_weights_diag=np.array([1.0, 0.0]),
        filling_tol=1e-10,
    )
    expected_mu = brentq(
        lambda mu: 0.5
        * (
            1
            - (0.5 - mu)
            / np.hypot(0.5 - mu, 0.3)
            * np.tanh(np.hypot(0.5 - mu, 0.3) / 0.4)
        )
        - 0.3,
        -3,
        3,
    )
    assert_allclose(result.mu, expected_mu, atol=1e-9)
    assert len(calls) > 3
    assert any(not np.array_equal(calls[0], matrices) for matrices in calls[1:])
    assert result.statistics.spectrum_bytes == 0


def test_adaptive_bdg_fixed_mu():
    hamiltonian = {
        (0,): np.array([[0.2, 0.3], [0.3, -0.2]]),
        (1,): np.diag([0.7, -0.7]),
        (-1,): np.diag([0.7, -0.7]),
    }
    kwargs = dict(
        keys=[(0,)],
        q_diag=np.array([1.0, -1.0]),
        trace_weights_diag=np.array([1.0, 0.0]),
        mu=0.17,
    )
    adaptive = evaluate(hamiltonian, **kwargs)
    reference = evaluate(hamiltonian, integration=PeriodicGrid(nk=16384), **kwargs)
    assert_allclose(adaptive.density.values, reference.density.values, atol=2e-6)
    assert adaptive.statistics.validation_evaluations > 0


def test_zero_temperature_fixed_grid_and_unattainable_filling():
    fixed = evaluate(kT=0, integration=PeriodicGrid(nk=16), mu=0.2)
    assert fixed.errors.density_matrix_integration is None
    with pytest.raises(RuntimeError, match="Chemical-potential solve failed"):
        evaluate(kT=0, integration=PeriodicGrid(nk=8), filling=0.37)
    with pytest.raises(ValueError, match="requires kT > 0"):
        evaluate(kT=0, mu=0)


@pytest.mark.parametrize(
    "integration,message",
    [
        (PeriodicGrid(nk=17, max_points=16), "total-grid-size"),
        (PeriodicGrid(nk=9, max_spectrum_bytes=64), "spectrum-storage"),
        (PeriodicGrid(max_spectrum_bytes=80), "spectrum-storage"),
    ],
)
def test_grid_and_spectrum_limits(integration, message):
    with pytest.raises(RuntimeError, match=message):
        evaluate(integration=integration, filling=0.37)


def test_global_root_budget():
    with pytest.raises(
        RuntimeError, match="max_charge_evaluations|charge-evaluation budget"
    ):
        evaluate(filling=0.37, max_charge_evaluations=4)


def test_sparse_defaults_never_silently_densify():
    from scipy.sparse import csr_matrix

    hamiltonian = {(0,): csr_matrix(np.eye(2))}
    with pytest.raises(ValueError, match="Automatic sparse"):
        resolve_periodic_matrix_function(None, hamiltonian, kT=0.2, prescribed=False)
    assert isinstance(
        resolve_periodic_matrix_function(None, hamiltonian, kT=0.2, prescribed=True),
        RationalFOE,
    )
    assert isinstance(
        resolve_periodic_matrix_function(
            DirectDiagonalization(), hamiltonian, kT=0.2, prescribed=False
        ),
        DirectDiagonalization,
    )
    with pytest.raises(ValueError, match="adaptive RationalFOE is unsupported"):
        resolve_periodic_matrix_function(
            RationalFOE(), hamiltonian, kT=0.2, prescribed=False
        )


def test_charge_derivative_matches_finite_difference_on_retained_grid():
    from meanfi.density.integrate.periodic import _Evaluator, _Grid
    from meanfi.errors import default_solver_tolerances
    from meanfi.space.coordinates import full_density_coordinates

    evaluator = _Evaluator(
        wire(),
        kT=0.2,
        integration=PeriodicGrid(nk=32),
        coordinates=full_density_coordinates([(0,)], size=1),
        q_diag=None,
        trace_weights=np.ones(1),
        tolerances=default_solver_tolerances(1e-4),
    )
    grid = _Grid(32, 1)
    evaluator.retain_spectra(grid, None)
    derivative = evaluator.charge(grid, 0.31)[2]
    finite_difference = (
        evaluator.charge(grid, 0.310001)[0] - evaluator.charge(grid, 0.309999)[0]
    ) / 2e-6
    assert_allclose(derivative, finite_difference, rtol=1e-7)


def test_normal_root_cost_does_not_grow_with_repeated_identical_points():
    calls = []
    for n in (4, 32, 256):
        result = evaluate(
            {(0,): np.array([[0.3]])},
            keys=[(0,)],
            integration=PeriodicGrid(nk=n),
            filling=0.37,
            filling_tol=1e-10,
        )
        assert_allclose(result.mu, 0.3 + 0.2 * np.log(0.37 / 0.63), atol=1e-9)
        calls.append(result.statistics.charge_evaluations)
    assert max(calls) <= 12
    assert max(calls) - min(calls) <= 1


def test_shifted_grid_rejects_diagonal_multidimensional_alias():
    hamiltonian = {
        (0, 0): np.zeros((1, 1)),
        (8, -8): np.ones((1, 1)),
        (-8, 8): np.ones((1, 1)),
    }
    # Equal shifts in both axes leave this entire oscillation invisible.
    with pytest.raises(RuntimeError, match="max_refinements=1"):
        evaluate(
            hamiltonian,
            keys=[(0, 0)],
            mu=0.3,
            integration=PeriodicGrid(max_refinements=1),
        )


@pytest.mark.parametrize("bdg", [False, True])
def test_sparse_aaa_reuses_one_scalar_fit_and_preserves_mu_dependence(monkeypatch, bdg):
    import scipy.sparse as sparse
    import meanfi.density.integrate.periodic as periodic
    import meanfi.density.kpoint.matrix_functions.rational.prepared_sparse as prepared

    hamiltonian = {(0,): sparse.csr_matrix([[0.4, 0.15], [0.15, -0.4]])}
    kwargs = dict(keys=[(0,)])
    if bdg:
        kwargs.update(
            q_diag=np.array([1.0, -1.0]), trace_weights_diag=np.array([1.0, 0.0])
        )
    fits = 0
    original_fit = prepared._aaa_terms_for_interval
    original_node = periodic.PreparedMumpsRationalNode
    cache_sizes = []

    def fit(*args, **kwargs):
        nonlocal fits
        fits += 1
        return original_fit(*args, **kwargs)

    def node(*args, **kwargs):
        cache_sizes.append(len(kwargs["shared_aaa_interval_cache"]))
        return original_node(*args, **kwargs)

    monkeypatch.setattr(prepared, "_aaa_terms_for_interval", fit)
    monkeypatch.setattr(periodic, "PreparedMumpsRationalNode", node)
    sparse_method = PeriodicGrid(
        nk=8, matrix_function=RationalFOE(rational_scheme="aaa")
    )
    dense_method = PeriodicGrid(nk=8, matrix_function=DirectDiagonalization())
    for mu in (0.1, 0.5):
        result = evaluate(hamiltonian, integration=sparse_method, mu=mu, **kwargs)
        reference = evaluate(hamiltonian, integration=dense_method, mu=mu, **kwargs)
        assert_allclose(result.density.values, reference.density.values, atol=2e-6)
        assert_allclose(result.filling, reference.filling, atol=2e-6)
    # Eight identical points need one scalar approximation for each new solve.
    assert fits == 2
    assert max(cache_sizes) == 1

    result = evaluate(
        hamiltonian, integration=sparse_method, filling=0.3 if bdg else 0.7, **kwargs
    )
    reference = evaluate(
        hamiltonian, integration=dense_method, filling=0.3 if bdg else 0.7, **kwargs
    )
    assert_allclose(result.mu, reference.mu, atol=2e-5)
    assert_allclose(result.density.values, reference.density.values, atol=2e-6)
    assert max(cache_sizes) == 1


@pytest.mark.parametrize("temperature", [np.nan, np.inf, -0.1])
def test_periodic_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="kT must be finite and non-negative"):
        evaluate(kT=temperature, integration=PeriodicGrid(nk=4), mu=0.1)


def test_explicit_filling_tolerance_controls_sparse_pointwise_accuracy():
    from scipy.sparse import csr_matrix
    from meanfi.errors import default_solver_tolerances

    result = evaluate(
        {(0,): csr_matrix((4, 4), dtype=complex)},
        keys=[(0,)],
        kT=0.2,
        filling=1.0,
        filling_tol=1e-6,
        tolerances=default_solver_tolerances(1e-3),
        integration=PeriodicGrid(nk=2),
        q_diag=np.array([1.0, 1.0, -1.0, -1.0]),
        trace_weights_diag=np.array([1.0, 1.0, 0.0, 0.0]),
    )
    assert result.statistics.n_diagonalizations is None
    # Check the physical charge independently of the rational approximation.
    assert abs(2 * expit(result.mu / 0.2) - 1) <= 1e-6
