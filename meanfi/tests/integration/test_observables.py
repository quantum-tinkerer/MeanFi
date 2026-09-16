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
    evaluate_free_energy,
    evaluate_internal_energy,
)
from meanfi.meanfield import (
    interaction_correction,
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
    assert evaluate_internal_energy(model, density) == pytest.approx(
        evaluate_internal_energy(model, matrix)
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


@pytest.mark.parametrize("density_format", ["result", "dense", "sparse"])
@pytest.mark.parametrize("sparse_observable", [False, True])
def test_expectation_value_requires_matching_matrix_sizes(
    density_format, sparse_observable
):
    matrix = np.array([[0.6, 0.1j], [-0.1j, 0.4]])
    density = density_result_from_tb({(): matrix})
    if density_format != "result":
        density = density.to_tb(sparse=density_format == "sparse")

    def observable(block):
        return {(): sparse.csr_matrix(block) if sparse_observable else block}

    operator = np.array([[2.0, 0.3j], [-0.3j, -1.0]])
    error = abs(expectation_value(density, observable(operator)) - 0.86)
    assert error < 1e-14, f"Observable contraction error: {error}"

    for block in (np.ones((1, 1)), np.ones((1, 2)), np.zeros((3, 3))):
        with pytest.raises(ValueError, match="must all have shape"):
            expectation_value(density, observable(block))


@pytest.mark.parametrize("use_sparse", [False, True])
def test_expectation_value_rejects_inconsistent_density_block_sizes(use_sparse):
    density = {(0,): np.eye(2), (1,): np.ones((1, 1))}
    if use_sparse:
        density = {key: sparse.csr_matrix(block) for key, block in density.items()}
    with pytest.raises(ValueError, match="must all have shape"):
        expectation_value(density, {(-1,): np.eye(2)})


def test_internal_energy_half_counts_normal_mean_field_interaction():
    model = Model(
        {(): np.diag([1.0, 3.0])},
        {(): np.array([[0.0, 2.0], [2.0, 0.0]], dtype=complex)},
        filling=1.0,
        kT=0.2,
    )
    density = {(): np.diag([0.25, 0.75]).astype(complex)}

    correction = model.mean_field(density)
    interaction_energy = expectation_value(density, correction)
    expected = expectation_value(density, model.h_0) + 0.5 * interaction_energy
    naive = expectation_value(density, add_tb(model.h_0, correction))

    assert evaluate_internal_energy(model, density) == pytest.approx(
        np.real(expected) / 2
    )
    assert naive / 2 - evaluate_internal_energy(model, density) == pytest.approx(
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
    error = abs(evaluate_internal_energy(model, density) - expected.real / 2)
    assert error < 1e-14, f"Reference-subtracted energy error: {error}"

    # Its derivative must be the Hamiltonian used for the density calculation.
    direction = np.array([[0.2, 0.13 - 0.09j], [0.13 + 0.09j, -0.2]])
    step = 1e-5
    numerical = (
        evaluate_internal_energy(model, {(): density[()] + step * direction})
        - evaluate_internal_energy(model, {(): density[()] - step * direction})
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
        evaluate_internal_energy(model, {(0,): np.array([[0.5]], dtype=complex)})


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
        evaluate_internal_energy(model, shifted(epsilon))
        - evaluate_internal_energy(model, shifted(-epsilon))
    ) / (2.0 * epsilon)
    rhs = np.real(expectation_value(direction, model.hamiltonian_from_density(rho)))

    assert evaluate_internal_energy(model, rho) == pytest.approx(slater_energy() / 4)
    assert derivative == pytest.approx(rhs / 4, rel=1e-8, abs=1e-8)


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

    assert correction[(1,)][0, 1] == pytest.approx(0.20000002)
    assert correction[(-1,)][0, 1] == pytest.approx(-0.20000002)


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
    assert evaluate_internal_energy(model, as_tb(rotated)) == pytest.approx(
        evaluate_internal_energy(model, as_tb(density)), abs=1e-14
    )
    epsilon = 1e-5
    derivative = (
        evaluate_internal_energy(model, as_tb(density + epsilon * direction))
        - evaluate_internal_energy(model, as_tb(density - epsilon * direction))
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
    assert evaluate_internal_energy(model, selected) == pytest.approx(
        evaluate_internal_energy(model, full)
    )
    assert evaluate_free_energy(model, selected) == pytest.approx(
        evaluate_internal_energy(model, full) - kT * entropy
    )
    assert selected.entropy == full.entropy


def test_free_energy_rejects_dictionary_without_entropy():
    model = Model({(): np.eye(1)}, {(): np.zeros((1, 1))}, filling=0.5, kT=0.2)
    with pytest.raises(TypeError, match="DensityResult with computed entropy"):
        evaluate_free_energy(model, {(): np.array([[0.5]])})


@pytest.mark.parametrize("bdg", [False, True])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_model_selected_density_keeps_physical_energy_without_one_body_entries(
    bdg,
    use_sparse,
    request,
):
    import meanfi as mf
    from scipy import sparse
    from scipy.special import expit

    h0 = np.array([[-0.4, 0.15j], [-0.15j, 0.3]])
    interaction = np.array([[0, 0.25], [0.25, 0]])
    if use_sparse:
        request.getfixturevalue("require_mumps")
        h0, interaction = sparse.csr_matrix(h0), sparse.csr_matrix(interaction)
    model = mf.Model({(): h0}, {(): interaction}, 1, kT=0.2, superconducting=bdg)
    integration = mf.UniformGrid()
    correction = model.random_meanfield(rng=17, scale=0.1)
    result = mf.density_matrix_at_mu(
        model,
        mu=0.07,
        mean_field=correction,
        integration=integration,
        tol=1e-9,
        compute_free_energy=True,
    )
    h = model.hamiltonian_from_meanfield(correction)[()]
    h = h.toarray() if use_sparse else h
    q = np.r_[np.ones(2), -np.ones(2)] if bdg else np.ones(2)
    energies, vectors = np.linalg.eigh(h - 0.07 * np.diag(q))
    exact = (vectors * expit(-energies / model.kT)) @ vectors.conj().T
    reference = mf.evaluate_internal_energy(model, {(): exact})
    assert result.internal_energy == pytest.approx(reference, abs=2e-9)
    assert result.free_energy == result.internal_energy - model.kT * result.entropy
    assert result.kT == model.kT
    assert result.coordinates is model.required_coordinates

    # Observable scalars describe the evaluated state, even after selection.
    empty = result.select(
        mf.DensityCoordinates.from_entries(
            size=len(exact),
            keys=[()],
            entries=(),
        )
    )
    assert empty.internal_energy == result.internal_energy
    assert empty.free_energy == result.free_energy


def test_default_selected_density_energy_does_not_require_hopping_entries():
    import meanfi as mf

    model = mf.Model(
        {(): np.array([[0.0, 0.2], [0.2, 0.0]])},
        {(): np.eye(2)},
        1,
        kT=0.2,
    )
    result = mf.density_matrix(model)
    assert result.coordinates.value_count == 2
    expected = -0.1 * np.tanh(0.5)
    for equivalent_model in (model, replace(model)):
        with pytest.raises(ValueError, match="missing"):
            mf.evaluate_internal_energy(equivalent_model, result)
    assert result.internal_energy == pytest.approx(expected, abs=1e-14)


def test_model_physical_inputs_have_one_owner():
    import meanfi as mf
    from dataclasses import replace

    model = mf.Model({(): np.diag([-0.4, 0.3])}, {(): np.zeros((2, 2))}, 1, kT=0.2)
    with pytest.raises(ValueError, match="set kT on the Model"):
        mf.density_matrix_at_mu(model, 0, kT=0.4)
    with pytest.raises(ValueError, match="set filling on the Model"):
        mf.density_matrix(model, filling=0.7)
    updated = replace(model, kT=0.4, filling=0.7)
    result = mf.density_matrix(updated, compute_free_energy=True)
    assert result.kT == 0.4
    assert result.filling == pytest.approx(0.7, abs=1e-4)
    assert result.free_energy == result.internal_energy - 0.4 * result.entropy


@pytest.mark.usefixtures("require_mumps")
@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("covered", [False, True])
def test_fixed_mu_selected_energy_uses_only_available_entries(
    monkeypatch, superconducting, covered
):
    import meanfi as mf
    from meanfi.density.kpoint.matrix_functions.rational import (
        PreparedMumpsRationalNode,
    )

    h0 = np.diag([-0.4, 0.3]).astype(complex)
    if not covered:
        h0[0, 1], h0[1, 0] = 0.07j, -0.07j
    model = mf.Model(
        {(): sparse.csr_matrix(h0)},
        {(): sparse.eye(2, format="csr")},
        filling=1,
        kT=0.2,
        superconducting=superconducting,
    )
    hamiltonian = model.hamiltonian_from_meanfield()[()].toarray()
    energies, vectors = np.linalg.eigh(hamiltonian)
    exact_density = (vectors * expit(-energies / model.kT)) @ vectors.conj().T
    expected = mf.evaluate_internal_energy(model, {(): exact_density})

    def unnecessary_work(*args, **kwargs):
        raise AssertionError("Model energy must use the requested density entries")

    monkeypatch.setattr(PreparedMumpsRationalNode, "charge", unnecessary_work)
    monkeypatch.setattr(PreparedMumpsRationalNode, "thermodynamics", unnecessary_work)
    result = mf.density_matrix_at_mu(model, 0, integration=mf.UniformGrid(), tol=1e-9)
    assert result.coordinates.value_count == 2
    assert not result.is_complete
    assert result.band_energy is result.entropy is result.free_energy is None
    if covered:
        assert result.internal_energy == pytest.approx(expected, abs=1e-9)
        assert result.internal_energy == mf.evaluate_internal_energy(model, result)
    else:
        assert result.internal_energy is None
        with pytest.raises(ValueError, match="missing"):
            mf.evaluate_internal_energy(model, result)


@pytest.mark.parametrize("use_sparse", [False, True])
def test_complex_density_has_creation_index_second_and_trace_observables(use_sparse):
    import meanfi as mf

    # The occupied state is (1, i)/sqrt(2); <c_1^dagger c_0> = -i/2.
    h = np.array([[0, 1j], [-1j, 0]])
    occupied = np.array([1, 1j]) / np.sqrt(2)
    expected = np.outer(occupied, occupied.conj())
    block = sparse.csr_matrix(h) if use_sparse else h
    density = mf.density_matrix_at_mu(
        {(): block},
        0,
        keys=[()],
        integration=mf.UniformGrid(matrix_function=mf.DirectDiagonalization()),
    )
    error = np.max(abs(density.to_tb()[()] - expected))
    assert error < 1e-14, f"Complex density convention error: {error}"
    assert density.to_tb()[()][0, 1] == pytest.approx(-0.5j)
    assert expectation_value(density, {(): block}) == pytest.approx(-1.0, abs=1e-14)
