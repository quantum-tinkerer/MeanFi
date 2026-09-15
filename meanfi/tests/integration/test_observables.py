from meanfi.results import _DensityEntries
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.special import entr, expit

from meanfi.tests.fixtures.models import density_result_from_tb

from meanfi import (
    DensityCoordinates,
    DensityResult,
    ErrorValues,
    Model,
    add_tb,
    expectation_value,
    free_energy,
    internal_energy,
)
from meanfi.meanfield import (
    interaction_correction,
    meanfield,
)
from meanfi.tests.fixtures.models import bipartite_hubbard_2d
from meanfi.tb.bdg import assemble_bdg_tb


pytestmark = pytest.mark.integration


def test_selected_density_supports_covered_observables_and_internal_energy():
    model = Model(
        {(): np.diag([1.0, 3.0]).astype(complex)},
        {(): np.array([[0.0, 2.0], [2.0, 0.0]], dtype=complex)},
        filling=1.0,
    )
    matrix = {(): np.array([[0.25, 0.1], [0.1, 0.75]], dtype=complex)}
    coordinates = model.required_coordinates
    density = DensityResult(
        entries=_DensityEntries(coordinates, coordinates.values_from_tb(matrix)),
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )

    assert density.is_complete is False
    assert expectation_value(density, model.h_0) == pytest.approx(
        expectation_value(matrix, model.h_0)
    )
    assert internal_energy(model, density) == pytest.approx(
        internal_energy(model, matrix)
    )


def test_selected_density_rejects_uncovered_observable_coordinate():
    coordinates = DensityCoordinates.from_entries(
        size=2,
        keys=[()],
        entries=(((), 0, 0),),
    )
    density = DensityResult(
        entries=_DensityEntries(coordinates, np.array([0.25])),
        mu=0.0,
        filling=0.25,
        errors=ErrorValues(),
    )

    with pytest.raises(ValueError, match="required by the observable"):
        expectation_value(density, {(): np.eye(2, dtype=complex)})


def test_internal_energy_half_counts_normal_mean_field_interaction():
    model = Model(
        {(): np.diag([1.0, 3.0])},
        {(): np.array([[0.0, 2.0], [2.0, 0.0]], dtype=complex)},
        filling=1.0,
        kT=0.2,
    )
    density = {(): np.diag([0.25, 0.75]).astype(complex)}

    correction = meanfield(density, model.h_int)
    interaction_energy = expectation_value(density, correction)
    expected = expectation_value(density, model.h_0) + 0.5 * interaction_energy
    naive = expectation_value(density, add_tb(model.h_0, correction))

    assert internal_energy(model, density) == pytest.approx(np.real(expected) / 2)
    assert naive / 2 - internal_energy(model, density) == pytest.approx(
        0.5 * interaction_energy / 2
    )


def test_internal_energy_uses_reference_subtracted_interaction_functional():
    h_0 = {(): np.diag([0.2, -0.3]).astype(complex)}
    interaction = 1.7
    h_int = {(): np.array([[0.0, interaction], [interaction, 0.0]])}
    reference = {(): np.array([[0.6, 0.11 + 0.04j], [0.11 - 0.04j, 0.4]])}
    density = {(): np.array([[0.25, 0.08 - 0.03j], [0.08 + 0.03j, 0.75]])}
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        reference=density_result_from_tb(reference),
    )
    # Wick's formula for two orbitals: V * (delta_n0 * delta_n1 - |delta_c|**2).
    # Check the scalar expression independently of the mean-field implementation.
    delta = density[()] - reference[()]
    expected = 0.2 * 0.25 - 0.3 * 0.75
    expected += interaction * (delta[0, 0] * delta[1, 1] - abs(delta[0, 1]) ** 2)
    error = abs(internal_energy(model, density) - expected.real / 2)
    assert error < 1e-14, f"Reference-subtracted energy error: {error}"

    # Its derivative must be the Hamiltonian used for the density calculation.
    direction = np.array([[0.2, 0.13 - 0.09j], [0.13 + 0.09j, -0.2]])
    step = 1e-5
    numerical = (
        internal_energy(model, {(): density[()] + step * direction})
        - internal_energy(model, {(): density[()] - step * direction})
    ) / (2 * step)
    h = model.hamiltonian_from_density(density)[()]
    analytic = np.trace(h @ direction).real / 2
    error = abs(numerical - analytic)
    # A centered difference is exact for this quadratic, up to roundoff / step.
    assert error < 1e-10, f"Reference-subtracted energy derivative error: {error}"


def test_internal_energy_rejects_missing_one_body_density_keys():
    model = Model(
        {
            (0,): np.zeros((1, 1), dtype=complex),
            (1,): np.ones((1, 1), dtype=complex),
            (-1,): np.ones((1, 1), dtype=complex),
        },
        {(0,): np.zeros((1, 1), dtype=complex)},
        filling=0.5,
    )

    with pytest.raises(ValueError, match="missing keys required by the observable"):
        internal_energy(model, {(0,): np.array([[0.5]], dtype=complex)})


def test_internal_energy_gradient_matches_hubbard_mean_field_hamiltonian():
    model = Model(*bipartite_hubbard_2d(U=3.7), filling=2.0, kT=0.2)
    rng = np.random.default_rng(1123)
    occupied, _ = np.linalg.qr(
        rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    )
    zero = np.zeros((4, 4), dtype=complex)
    rho = {key: np.array(zero, copy=True) for key in model.h_0}
    rho[(0, 0)] = occupied @ occupied.conj().T
    raw_direction = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    direction = {key: np.array(zero, copy=True) for key in model.h_0}
    direction[(0, 0)] = raw_direction + raw_direction.conj().T
    epsilon = 1e-6

    def shifted(scale):
        return {key: rho[key] + scale * direction[key] for key in rho}

    def slater_energy():
        h_0 = model.h_0[(0, 0)]
        interaction = model.h_int[(0, 0)]
        first, second = occupied.T
        vec = (np.kron(first, second) - np.kron(second, first)) / np.sqrt(2.0)
        one_body = np.kron(h_0, np.eye(4)) + np.kron(np.eye(4), h_0)
        two_body = np.diag(interaction.ravel())
        return float(np.real(np.vdot(vec, (one_body + two_body) @ vec)))

    derivative = (
        internal_energy(model, shifted(epsilon))
        - internal_energy(model, shifted(-epsilon))
    ) / (2.0 * epsilon)
    rhs = np.real(expectation_value(direction, model.hamiltonian_from_density(rho)))

    assert internal_energy(model, rho) == pytest.approx(slater_energy() / 4)
    assert derivative == pytest.approx(rhs / 4, rel=1e-8, abs=1e-8)


def test_internal_energy_matches_bdg_block_formula():
    model = Model(
        {(): np.diag([2.0, 3.0]).astype(complex)},
        {(): np.array([[0.0, 1.5], [1.5, 0.0]], dtype=complex)},
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )
    density = {
        (): np.array(
            [
                [0.4, 0.05, 0.0, 0.2],
                [0.05, 0.3, -0.2, 0.0],
                [0.0, -0.2, 0.6, 0.01],
                [0.2, 0.0, 0.01, 0.7],
            ],
            dtype=complex,
        )
    }

    # Independent two-orbital Wick expression, including attractive pairing.
    expected = 2.0 * 0.4 + 3.0 * 0.3 + 1.5 * (0.4 * 0.3 - 0.05**2 - 0.2**2)
    assert internal_energy(model, density) == pytest.approx(expected / 2)


def test_bdg_correction_projects_pairing_antisymmetry_noise():
    model = Model(
        {
            (0,): np.zeros((1, 1), dtype=complex),
            (1,): np.zeros((1, 1), dtype=complex),
            (-1,): np.zeros((1, 1), dtype=complex),
        },
        {
            (1,): np.ones((1, 1), dtype=complex),
            (-1,): np.ones((1, 1), dtype=complex),
        },
        filling=0.5,
        kT=0.1,
        superconducting=True,
    )
    density = {
        (0,): np.zeros((2, 2), dtype=complex),
        (1,): np.array([[0.0, 0.2], [0.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, -0.20000004], [0.0, 0.0]], dtype=complex),
    }

    correction = interaction_correction(density, model.h_int, electron_ndof=model._ndof)

    assert correction[(1,)][0, 1] == pytest.approx(-0.20000002)
    assert correction[(-1,)][0, 1] == pytest.approx(0.20000002)


@pytest.mark.parametrize("use_sparse", [False, True])
def test_bdg_energy_is_phase_invariant_and_has_the_hamiltonian_gradient(
    use_sparse, monkeypatch
):
    h_0 = {(): np.array([[0.3, 0.04j], [-0.04j, -0.2]])}
    h_int = {(): np.array([[0.0, 1.5], [1.5, 0.0]])}
    if use_sparse:
        h_0 = {key: sparse.csr_matrix(block) for key, block in h_0.items()}
        h_int = {key: sparse.csr_matrix(block) for key, block in h_int.items()}
    model = Model(h_0, h_int, filling=1.0, kT=0.2, superconducting=True)
    trial = assemble_bdg_tb(
        {(): np.array([[0.48, 0.06j], [-0.06j, -0.27]])},
        {(): np.array([[0.0, 0.2 + 0.13j], [-0.2 - 0.13j, 0.0]])},
        ndof=2,
    )[()]
    energies, vectors = np.linalg.eigh(trial - 0.11 * np.diag([1, 1, -1, -1]))
    density = (vectors * expit(-energies / model.kT)) @ vectors.conj().T
    direction = assemble_bdg_tb(
        {(): np.array([[0.02, 0.05j], [-0.05j, -0.03]])},
        {(): np.array([[0.0, 0.017 + 0.03j], [-0.017 - 0.03j, 0.0]])},
        ndof=2,
    )[()]
    rotated = density.copy()
    rotated[:2, 2:] *= np.exp(0.74j)
    rotated[2:, :2] *= np.exp(-0.74j)

    def as_tb(block):
        return {(): sparse.csr_matrix(block) if use_sparse else block}

    if use_sparse:

        def forbid_dense(*args, **kwargs):
            raise AssertionError("sparse energy evaluation must not densify blocks")

        monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
    assert internal_energy(model, as_tb(rotated)) == pytest.approx(
        internal_energy(model, as_tb(density)), abs=1e-14
    )
    epsilon = 1e-5
    derivative = (
        internal_energy(model, as_tb(density + epsilon * direction))
        - internal_energy(model, as_tb(density - epsilon * direction))
    ) / (2 * epsilon)
    expected = (
        0.5
        * expectation_value(
            as_tb(direction), model.hamiltonian_from_density(as_tb(density))
        ).real
    )
    assert derivative == pytest.approx(expected / 2, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("kT", [0.0, 0.2])
def test_free_energy_uses_full_state_entropy_after_selecting_entries(
    superconducting, kT
):
    model = Model(
        {(): np.diag([0.2, -0.3])},
        {(): np.array([[0.0, 1.7], [1.7, 0.0]])},
        filling=1.0,
        kT=kT,
        superconducting=superconducting,
    )
    normal = np.diag([0.25, 0.75]).astype(complex)
    if superconducting:
        pairing = np.array([[0.0, 0.1j], [-0.1j, 0.0]])
        matrix = np.block([[normal, pairing], [pairing.conj().T, np.eye(2) - normal.T]])
    else:
        matrix = normal
    occupations = np.linalg.eigvalsh(matrix)
    entropy = np.sum(entr(occupations) + entr(1 - occupations))
    if superconducting:
        entropy *= 0.5
    entropy /= 2
    full = replace(density_result_from_tb({(): matrix}), entropy=float(entropy))
    selected = full.select(model.required_coordinates)

    assert not selected.is_complete
    assert internal_energy(model, selected) == pytest.approx(
        internal_energy(model, full)
    )
    assert free_energy(model, selected) == pytest.approx(
        internal_energy(model, full) - kT * entropy
    )
    assert selected.entropy == full.entropy


def test_free_energy_rejects_dictionary_without_entropy():
    model = Model({(): np.eye(1)}, {(): np.zeros((1, 1))}, filling=0.5, kT=0.2)
    with pytest.raises(TypeError, match="DensityResult with computed entropy"):
        free_energy(model, {(): np.array([[0.5]])})
