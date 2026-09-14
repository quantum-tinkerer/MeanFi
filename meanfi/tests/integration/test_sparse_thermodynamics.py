import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import expit, xlogy

from meanfi import RationalFOE
from meanfi.density.kpoint.matrix_functions.mumps_backend import (
    SelectedInverseFactorization,
)
from meanfi.density.kpoint.matrix_functions.rational import PreparedMumpsRationalNode
from meanfi.density.kpoint.matrix_functions.rational.scheme import (
    _aaa_terms_for_interval,
    thermal_errors,
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
        trace_weights_diag=trace_weights,
        options=RationalFOE(max_poles=128),
        charge_tolerance=1e-9,
        density_tolerance=1e-9,
        thermodynamic_tolerance=1e-8,
        density_coordinates=coordinates,
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
    assert node._charge_pattern.diagonal_positions.size == matrix.shape[0]


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
        thermodynamic_tolerance=1e-10,
        density_coordinates=coords,
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
        thermodynamic_tolerance=1e-8,
        density_coordinates=coordinates,
    )
    _, expected, _ = _reference(matrix, np.ones(4), kT, mu)
    node.charge(mu)
    actual_energy, _ = node.thermodynamics(mu)
    np.testing.assert_allclose(actual_energy, expected, atol=1e-8, rtol=0)


def test_thermal_targets_handle_extreme_energy_over_temperature():
    from meanfi.density.kpoint.matrix_functions.rational.scheme import thermal_targets

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
        thermodynamic_tolerance=tolerance,
        density_coordinates=coordinates,
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
