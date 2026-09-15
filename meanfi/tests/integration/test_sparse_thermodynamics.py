import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import expit, xlogy

from meanfi import RationalFOE
from meanfi.density.kpoint.matrix_functions.mumps_backend import (
    SelectedInverseFactorization,
)
from meanfi.density.kpoint.matrix_functions.rational import PreparedMumpsRationalNode
from meanfi.density.kpoint.matrix_functions.rational.common import SparseRationalLayout
from meanfi.density.kpoint.matrix_functions.rational.scheme import (
    _aaa_terms_for_interval,
    thermal_errors,
    thermal_targets,
)
from meanfi.space.coordinates import DensityCoordinates

pytestmark = pytest.mark.integration


def _reference(matrix, q_diag, kT, mu):
    energies, vectors = np.linalg.eigh(matrix - mu * np.diag(q_diag))
    occupations = expit(-energies / kT)
    density = (vectors * occupations) @ vectors.conj().T
    entropy = -np.sum(
        xlogy(occupations, occupations) + xlogy(1 - occupations, 1 - occupations)
    )
    return density, float(np.trace(matrix @ density).real), float(entropy)


@pytest.mark.parametrize("kT", [0.2, 0.01, 0.001])
def test_joint_aaa_certifies_density_and_entropy_on_independent_grid(kT):
    terms = _aaa_terms_for_interval(
        lower=-3,
        upper=3,
        kT=kT,
        scalar_tolerance=1e-9,
        entropy_tolerance=1e-9,
        initial_poles=4,
        pole_cap=100,
    )
    grid = np.unique(
        np.r_[np.linspace(-3, 3, 20001), np.linspace(-40 * kT, 40 * kT, 7000)]
    )
    grid = grid[(grid >= -3) & (grid <= 3)]
    f = expit(-grid / kT)
    entropy = -xlogy(f, f) - xlogy(1 - f, 1 - f)
    assert np.all(thermal_errors(terms, grid, np.column_stack([f, entropy])) < 1e-9)
    assert terms.entropy_residues.shape == terms.residues.shape == terms.shifts.shape


@pytest.mark.usefixtures("require_mumps")
@pytest.mark.parametrize("bdg", [False, True])
def test_sparse_thermodynamics_reuses_density_factorizations(monkeypatch, bdg):
    matrix = np.array(
        [
            [0.4, 0.2j, 0.12, 0],
            [-0.2j, -0.6, 0, -0.12],
            [0.12, 0, -0.4, 0.2j],
            [0, -0.12, -0.2j, 0.6],
        ],
        dtype=complex,
    )
    if bdg:
        # At a single k-point the hole trace need not complement the electron
        # trace; particle/hole symmetry relates k to its distinct -k partner.
        matrix[3, 3] += 0.17
    q_diag = np.array([1, 1, -1, -1]) if bdg else np.ones(4)
    trace_weights = np.array([1, 1, 0, 0]) if bdg else np.ones(4)
    mu, kT = 0.13, 0.2
    density, energy, entropy = _reference(matrix, q_diag, kT, mu)
    if bdg:
        assert (
            abs(
                q_diag @ density.diagonal()
                - (2 * trace_weights @ density.diagonal() - 2)
            )
            > 1e-4
        )
    coordinates = DensityCoordinates.from_entries(
        entries=(((), 0, 0), ((), 0, 1), ((), 1, 2)), size=4, keys=[()]
    )
    node = PreparedMumpsRationalNode(
        sp.csr_matrix(matrix),
        kT=kT,
        q_diag=q_diag,
        options=RationalFOE(max_poles=128),
        charge_tolerance=1e-9,
        density_tolerance=1e-9,
        band_energy_tolerance=1e-8,
        entropy_tolerance=1e-8,
        layout=SparseRationalLayout.build(
            density_coordinates=coordinates,
            trace_weights_diag=trace_weights,
            include_all_diagonal=True,
        ),
    )

    def no_dense_eigensolve(*args, **kwargs):
        raise AssertionError(
            "Sparse thermodynamics must not diagonalize the Hamiltonian"
        )

    monkeypatch.setattr(np.linalg, "eigh", no_dense_eigensolve)
    charge = node.charge(mu)
    np.testing.assert_allclose(charge, trace_weights @ density.diagonal(), atol=1e-9)
    factors = dict(node._last_factorizations)
    values = node.density_values_from_charge_order(mu)
    np.testing.assert_allclose(
        values, coordinates.values_from_assembled_matrix(density), atol=1e-9
    )

    def no_more_inverse_work(*args, **kwargs):
        raise AssertionError("Thermodynamics must reuse the charge inverse entries")

    monkeypatch.setattr(SelectedInverseFactorization, "factor", no_more_inverse_work)
    monkeypatch.setattr(
        SelectedInverseFactorization, "selected_inverse", no_more_inverse_work
    )
    actual_energy, actual_entropy = node.thermodynamics(mu)
    np.testing.assert_allclose(actual_energy, energy, atol=1e-8, rtol=0)
    np.testing.assert_allclose(actual_entropy, entropy, atol=1e-8, rtol=0)
    assert node._last_factorizations == factors
    assert node.layout.charge.nnz == matrix.shape[0]


@pytest.mark.usefixtures("require_mumps")
@pytest.mark.parametrize("mu", [-80, 0.11, 80])
def test_sparse_thermodynamics_handles_filled_empty_and_narrow_spectra(mu):
    matrix = np.diag([0.1, 0.1 + 1e-10, 0.1 + 2e-10]).astype(complex)
    coords = DensityCoordinates.from_entries(entries=(((), 0, 0),), size=3, keys=[()])
    node = PreparedMumpsRationalNode(
        sp.csr_matrix(matrix),
        kT=0.1,
        q_diag=np.ones(3),
        options=RationalFOE(),
        charge_tolerance=1e-10,
        density_tolerance=1e-10,
        band_energy_tolerance=1e-10,
        entropy_tolerance=1e-10,
        layout=SparseRationalLayout.build(
            density_coordinates=coords,
            trace_weights_diag=np.ones(coords.size),
            include_all_diagonal=True,
        ),
    )
    _, energy, entropy = _reference(matrix, np.ones(3), 0.1, mu)
    node.charge(mu)
    np.testing.assert_allclose(
        node.thermodynamics(mu), (energy, entropy), atol=1e-10, rtol=0
    )
    with pytest.raises(ValueError, match="Evaluate charge"):
        node.thermodynamics(mu + 1)


@pytest.mark.parametrize(
    "lower,upper,tolerances,pole_cap",
    [
        (-2.2218477871579956, 8.218921136460922, [1.52e-11, 1.25e-10], 128),
        (
            -2.7305826304965994,
            2.503516271744375,
            [1.0924347951653114e-14, 3.125e-14],
            256,
        ),
    ],
)
def test_joint_aaa_certifies_asymmetric_intervals_without_real_poles(
    lower, upper, tolerances, pole_cap
):
    kT = 0.02
    terms = _aaa_terms_for_interval(
        pole_cap,
        lower=lower,
        upper=upper,
        kT=kT,
        scalar_tolerance=tolerances[0],
        entropy_tolerance=tolerances[1],
    )
    grid = np.linspace(lower, upper, 50003)
    f = expit(-grid / kT)
    entropy = -xlogy(f, f) - xlogy(1 - f, 1 - f)
    # Independent grid evaluations can differ by a few ulps across BLAS builds.
    # Keep this roundoff allowance separate from the requested fit tolerance.
    roundoff = 4 * np.finfo(float).eps
    assert np.all(
        thermal_errors(terms, grid, np.column_stack([f, entropy]))
        <= np.asarray(tolerances) + roundoff
    )
    assert not np.any(
        (terms.shifts.imag == 0)
        & (terms.shifts.real >= lower)
        & (terms.shifts.real <= upper)
    )


@pytest.mark.usefixtures("require_mumps")
def test_sparse_band_energy_retains_accuracy_far_from_zero_energy():
    matrix = np.diag([999.5, 999.8, 1000.2, 1000.9]).astype(complex)
    matrix[0, 1] = 0.11j
    matrix[1, 0] = -0.11j
    mu, kT = 1000.13, 0.2
    coordinates = DensityCoordinates.from_entries(
        size=4, keys=[()], entries=(((), 0, 0),)
    )
    node = PreparedMumpsRationalNode(
        sp.csr_matrix(matrix),
        kT=kT,
        q_diag=np.ones(4),
        options=RationalFOE(),
        charge_tolerance=1e-7,
        density_tolerance=1e-7,
        band_energy_tolerance=1e-8,
        entropy_tolerance=1e-8,
        layout=SparseRationalLayout.build(
            density_coordinates=coordinates,
            trace_weights_diag=np.ones(coordinates.size),
            include_all_diagonal=True,
        ),
    )
    _, expected, _ = _reference(matrix, np.ones(4), kT, mu)
    node.charge(mu)
    actual_energy, _ = node.thermodynamics(mu)
    np.testing.assert_allclose(actual_energy, expected, atol=1e-8, rtol=0)


def test_thermal_targets_handle_extreme_energy_over_temperature():
    targets = thermal_targets(np.array([-1e308, 0.0, 1e308]), 1e-300)
    np.testing.assert_allclose(targets[:, 0], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(targets[:, 1], [0.0, np.log(2.0), 0.0])


@pytest.mark.parametrize("control", ["initial_poles", "max_poles"])
@pytest.mark.parametrize("value", [False, 0, -1, 4.5])
def test_rational_pole_budgets_require_positive_integers(control, value):
    with pytest.raises(ValueError, match=f"{control} must be a positive integer"):
        RationalFOE(**{control: value})


def test_rational_minimum_can_use_the_entire_pole_budget():
    options = RationalFOE(initial_poles=4, max_poles=4)
    terms = _aaa_terms_for_interval(
        options.max_poles,
        lower=0.1,
        upper=0.1,
        kT=0.2,
        initial_poles=options.initial_poles,
        scalar_tolerance=1e-8,
        entropy_tolerance=1e-8,
    )
    np.testing.assert_allclose(terms.constant, expit(-0.5), atol=1e-8)


@pytest.mark.usefixtures("require_mumps")
def test_joint_aaa_meets_original_tight_32_orbital_benchmark_tolerance():
    # This case previously stalled before residue refitting near machine precision.
    size, mu, kT, tolerance = 32, 0.13, 0.02, 1e-12
    rng = np.random.default_rng(174729 + size)
    diagonal = rng.uniform(-0.4, 0.4, size)
    hopping = -(1 + 0.15 * rng.random(size - 1)) * np.exp(0.17j)
    matrix = sp.diags([hopping, diagonal, hopping.conj()], [-1, 0, 1], format="csr")
    rows, cols = matrix.nonzero()
    coordinates = DensityCoordinates.from_pairs(
        size=size,
        keys=[()],
        pairs_by_key={(): (rows, cols)},
    )
    node = PreparedMumpsRationalNode(
        matrix,
        kT=kT,
        q_diag=np.ones(size),
        options=RationalFOE(),
        charge_tolerance=tolerance,
        density_tolerance=tolerance,
        band_energy_tolerance=tolerance,
        entropy_tolerance=tolerance,
        layout=SparseRationalLayout.build(
            density_coordinates=coordinates,
            trace_weights_diag=np.ones(coordinates.size),
            include_all_diagonal=True,
        ),
    )
    density, energy, entropy = _reference(matrix.toarray(), np.ones(size), kT, mu)
    assert node.charge(mu) == pytest.approx(
        np.trace(density).real, abs=tolerance, rel=0
    )
    np.testing.assert_allclose(
        node.density_values_from_charge_order(mu),
        coordinates.values_from_assembled_matrix(density),
        atol=tolerance,
        rtol=0,
    )
    np.testing.assert_allclose(
        node.thermodynamics(mu), [energy, entropy], atol=tolerance, rtol=0
    )


@pytest.mark.parametrize("bdg", [False, True])
def test_nearby_intervals_reuse_one_accurate_fit(bdg):
    from meanfi.tb.bdg import assemble_bdg_tb

    matrix = np.array([[-0.3, 0.05j], [-0.05j, 0.2]])
    if bdg:
        matrix = assemble_bdg_tb(
            {(): matrix}, {(): np.array([[0.0, 0.12], [-0.12, 0.0]])}, ndof=2
        )[()]
    size = len(matrix)
    q = np.array([1, 1, -1, -1]) if bdg else np.ones(size)
    weights = np.array([1, 1, 0, 0]) if bdg else np.ones(size)
    coordinates = DensityCoordinates.from_entries(size=size, keys=[()], entries=())
    cache = []

    def fit(mu, offset=0.0, **overrides):
        options = dict(
            kT=0.1,
            q_diag=q,
            options=RationalFOE(),
            charge_tolerance=1e-8,
            density_tolerance=1e-8,
            band_energy_tolerance=1e-8,
            entropy_tolerance=1e-8,
            layout=SparseRationalLayout.build(
                density_coordinates=coordinates,
                trace_weights_diag=weights,
                include_all_diagonal=True,
            ),
            shared_aaa_interval_cache=cache,
        )
        options.update(overrides)
        node = PreparedMumpsRationalNode(
            sp.csr_matrix(matrix + offset * np.diag(q)), **options
        )
        terms = node._sparse_terms(mu)
        # Check the whole interval independently, including the Fermi transition.
        energies = np.linspace(cache[0].lower, cache[0].upper, 15001)
        errors = thermal_errors(terms, energies, thermal_targets(energies, node.kT))
        assert np.all(errors < 1e-8)
        assert len(cache) == 1
        return terms

    first = fit(0.0)
    widened = fit(0.01)
    assert widened is not first
    assert fit(0.015) is widened
    assert fit(0.015, offset=0.005) is widened
    assert fit(0.015, kT=0.2) is not widened


def test_shared_fit_rechecks_accuracy_entropy_and_pole_budget():
    from meanfi import ConvergenceError

    coordinates = DensityCoordinates.from_entries(size=2, keys=[()], entries=())
    cache = []

    def node(tolerance, **overrides):
        options = dict(
            kT=0.02,
            q_diag=np.ones(2),
            options=RationalFOE(),
            charge_tolerance=tolerance,
            density_tolerance=tolerance,
            layout=SparseRationalLayout.build(
                density_coordinates=coordinates,
                trace_weights_diag=np.ones(coordinates.size),
                include_all_diagonal=False,
            ),
            shared_aaa_interval_cache=cache,
        )
        options.update(overrides)
        return PreparedMumpsRationalNode(sp.diags([-0.3, 0.2], format="csr"), **options)

    loose = node(0.1)._sparse_terms(0.0)
    tight = node(1e-10)._sparse_terms(0.0)
    assert tight is not loose
    assert tight.entropy_residues is None
    joint = node(
        1e-10, band_energy_tolerance=1e-10, entropy_tolerance=1e-10
    )._sparse_terms(0.0)
    assert joint.entropy_residues is not None
    grid = np.linspace(-0.3, 0.2, 15001)
    assert np.all(thermal_errors(joint, grid, thermal_targets(grid, 0.02)) < 5e-11)
    with pytest.raises(ConvergenceError, match="max_poles"):
        node(1e-10, options=RationalFOE(max_poles=4))._sparse_terms(0.0)


def test_failed_interval_expansion_retries_actual_spectrum(monkeypatch):
    from meanfi.density.kpoint.matrix_functions.rational import prepared_sparse

    coordinates = DensityCoordinates.from_entries(size=2, keys=[()], entries=())
    cache = []
    node = PreparedMumpsRationalNode(
        sp.diags([-0.3, 0.2], format="csr"),
        kT=0.1,
        q_diag=np.ones(2),
        options=RationalFOE(),
        charge_tolerance=1e-10,
        density_tolerance=1e-10,
        band_energy_tolerance=1e-10,
        entropy_tolerance=1e-10,
        layout=SparseRationalLayout.build(
            density_coordinates=coordinates,
            trace_weights_diag=np.ones(coordinates.size),
            include_all_diagonal=True,
        ),
        shared_aaa_interval_cache=cache,
    )
    node._sparse_terms(0.0)
    fit = prepared_sparse._aaa_terms_for_interval
    attempts = []

    def restricted(*args, **kwargs):
        attempts.append((kwargs["lower"], kwargs["upper"]))
        if kwargs["upper"] - kwargs["lower"] > 0.6:
            raise ValueError("expanded interval exceeds the available fit budget")
        return fit(*args, **kwargs)

    monkeypatch.setattr(prepared_sparse, "_aaa_terms_for_interval", restricted)
    terms = node._sparse_terms(0.01)
    assert len(attempts) == 2
    assert attempts[0][0] < attempts[1][0] < attempts[1][1] < attempts[0][1]
    grid = np.linspace(-0.31, 0.19, 15001)
    assert np.all(thermal_errors(terms, grid, thermal_targets(grid, 0.1)) < 5e-11)


def test_joint_aaa_refines_the_grid_for_a_sharp_fermi_transition():
    kT = 1e-5
    terms = _aaa_terms_for_interval(
        256,
        lower=-3.0,
        upper=2.0,
        kT=kT,
        scalar_tolerance=1e-10,
        entropy_tolerance=1e-9,
    )
    # Resolve both the narrow transition and the much wider spectral interval.
    grid = np.unique(
        np.concatenate(
            [np.linspace(-3, 2, 50003), np.linspace(-40 * kT, 40 * kT, 20003)]
        )
    )
    f = expit(-grid / kT)
    entropy = -xlogy(f, f) - xlogy(1 - f, 1 - f)
    assert np.all(
        thermal_errors(terms, grid, np.column_stack([f, entropy]))
        <= np.array([1e-10, 1e-9])
    )


@pytest.mark.usefixtures("require_mumps")
@pytest.mark.parametrize("include_all", [False, True])
def test_shared_sparse_layout_preserves_requested_complex_entries(include_all):
    # A complex BdG-like matrix exercises both inverse orientations, hole
    # diagonals absent from the charge trace, and repeated pairs at different R.
    matrix = np.array(
        [[0.3, 0.12j, 0.08], [-0.12j, -0.4, 0.1j], [0.08, -0.1j, 0.6]],
        dtype=complex,
    )
    coordinates = DensityCoordinates.from_entries(
        size=3,
        keys=[(0,), (1,)],
        entries=(((0,), 0, 1), ((0,), 1, 2), ((0,), 2, 2), ((1,), 0, 1)),
    )
    weights = np.array([1.0, 1.0, 0.0])
    q_diag = np.array([1.0, 1.0, -1.0])
    layout = SparseRationalLayout.build(
        density_coordinates=coordinates,
        trace_weights_diag=weights,
        include_all_diagonal=include_all,
    )
    # Read-only storage prevents one node from invalidating every other node's
    # coordinate contract. Construction also owns a copy of the input weights.
    weights[:] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        layout.charge_weights[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        layout.density.rows[0] = 2
    with pytest.raises(TypeError):
        layout.density.lookup[(0, 1)] = 0
    pattern_arrays = [
        (array, array.copy())
        for pattern in (layout.charge, layout.density, layout.extra)
        for array in (pattern.fortran_indptr, pattern.fortran_indices)
    ]
    for offset, mu in ((0.0, 0.12), (0.07, -0.16), (0.0, 0.12)):
        hamiltonian = matrix + np.diag([offset, -offset, offset])
        density, _, _ = _reference(hamiltonian, q_diag, 0.15, mu)
        node = PreparedMumpsRationalNode(
            sp.csr_matrix(hamiltonian),
            kT=0.15,
            q_diag=q_diag,
            options=RationalFOE(),
            charge_tolerance=1e-9,
            density_tolerance=1e-9,
            layout=layout,
        )
        assert node.layout is layout
        assert abs(node.charge(mu) - np.trace(density[:2, :2]).real) <= 1e-9
        np.testing.assert_allclose(
            node.density_values_from_charge_order(mu),
            coordinates.values_from_assembled_matrix(density),
            atol=1e-9,
            rtol=0,
        )
    for array, original in pattern_arrays:
        np.testing.assert_array_equal(array, original)


@pytest.mark.usefixtures("require_mumps")
@pytest.mark.parametrize(
    "energy_tolerance,entropy_tolerance", [(1e-9, 1e-5), (1e-5, 1e-9)]
)
def test_sparse_thermodynamic_targets_have_independent_units(
    energy_tolerance, entropy_tolerance
):
    matrix = np.diag([999.5, 999.8, 1000.2, 1000.9]).astype(complex)
    matrix[0, 1], matrix[1, 0] = 0.11j, -0.11j
    mu, kT = 1000.13, 0.2
    coordinates = DensityCoordinates.from_entries(size=4, keys=[()], entries=())
    node = PreparedMumpsRationalNode(
        sp.csr_matrix(matrix),
        kT=kT,
        q_diag=np.ones(4),
        options=RationalFOE(),
        charge_tolerance=1e-4,
        density_tolerance=1e-4,
        band_energy_tolerance=energy_tolerance,
        entropy_tolerance=entropy_tolerance,
        layout=SparseRationalLayout.build(
            density_coordinates=coordinates, trace_weights_diag=np.ones(4)
        ),
    )
    _, expected_energy, expected_entropy = _reference(matrix, np.ones(4), kT, mu)
    node.charge(mu)
    energy, entropy = node.thermodynamics(mu)
    assert abs(energy - expected_energy) <= energy_tolerance
    assert abs(entropy - expected_entropy) <= entropy_tolerance
    scalar_tolerances = node._scalar_tolerances(-1.0, 1.0, mu)
    assert scalar_tolerances[0] == pytest.approx(energy_tolerance / (4 * (1 + mu)))
    assert scalar_tolerances[1] == pytest.approx(entropy_tolerance / 4)


@pytest.mark.usefixtures("require_mumps")
@pytest.mark.parametrize("band_energy_tolerance", [None, 1e-8])
def test_cached_entropy_fit_does_not_enable_unrequested_thermodynamics(
    band_energy_tolerance,
):
    coordinates = DensityCoordinates.from_entries(size=4, keys=[()], entries=())
    matrix = sp.diags([-0.4, 0.3, 0.4, -0.3], format="csr")
    weights = np.array([1.0, 1.0, 0.0, 0.0])
    cache = []

    def node(*, energy, entropy):
        return PreparedMumpsRationalNode(
            matrix,
            kT=0.2,
            q_diag=np.array([1.0, 1.0, -1.0, -1.0]),
            options=RationalFOE(),
            charge_tolerance=1e-8,
            density_tolerance=1e-8,
            band_energy_tolerance=energy,
            entropy_tolerance=entropy,
            layout=SparseRationalLayout.build(
                density_coordinates=coordinates,
                trace_weights_diag=weights,
                include_all_diagonal=energy is not None or entropy is not None,
            ),
            shared_aaa_interval_cache=cache,
        )

    joint = node(energy=1e-8, entropy=1e-8)
    joint.charge(0.1)
    charge_only = node(energy=band_energy_tolerance, entropy=None)
    charge_only.charge(0.1)
    assert charge_only._last_terms is joint._last_terms
    with pytest.raises(ValueError, match="Prepare the node with entropy_tolerance"):
        charge_only.thermodynamics(0.1)
