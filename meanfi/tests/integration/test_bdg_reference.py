"""BdG references checked against two-orbital formulas and a scalar gap equation."""

from meanfi.results import _DensityEntries

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import brentq
from scipy.special import entr, expit

import meanfi as mf
from meanfi.tests.fixtures.models import density_result_from_tb

pytestmark = pytest.mark.integration


def covariance(normal, pairing):
    return np.block([[normal, pairing], [pairing.conj().T, np.eye(2) - normal.T]])


@pytest.mark.parametrize("normal_reference", [False, True])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_bdg_reference_energy_and_hamiltonian(
    normal_reference, selected, use_sparse, monkeypatch
):
    h0 = np.array([[0.3, 0.04j], [-0.04j, -0.2]])
    interaction = 1.5
    hint = np.array([[0.0, interaction], [interaction, 0.0]])
    normal = np.array([[0.48, 0.06j], [-0.06j, 0.52]])
    pairing = np.array([[0.0, 0.11 + 0.04j], [-0.11 - 0.04j, 0.0]])
    ref_normal = np.array([[0.55, 0.02 + 0.01j], [0.02 - 0.01j, 0.45]])
    ref_pairing = (
        np.zeros((2, 2), complex)
        if normal_reference
        else np.array([[0.0, 0.025 - 0.015j], [-0.025 + 0.015j, 0.0]])
    )
    reference = density_result_from_tb(
        {(): ref_normal if normal_reference else covariance(ref_normal, ref_pairing)}
    )
    raw = covariance(normal, pairing)
    eigenvalues = np.linalg.eigvalsh(raw)
    entropy = float(np.sum(entr(eigenvalues) + entr(1 - eigenvalues))) / 4
    density = replace(density_result_from_tb({(): raw}), entropy=entropy)

    def tb(matrix):
        return {(): sparse.csr_matrix(matrix) if use_sparse else matrix}

    bare = mf.Model(tb(h0), tb(hint), filling=1.0, kT=0.2, superconducting=True)
    if selected:
        ref_model = replace(bare, superconducting=not normal_reference)
        reference = reference.select(ref_model.required_coordinates)
        # The bare Hamiltonian also needs the conjugate off-diagonal entry.
        energy_coordinates = mf.DensityCoordinates.from_entries(
            size=4,
            keys=[()],
            entries=tuple(
                sorted(set(bare.required_coordinates.entries) | {((), 1, 0)})
            ),
        )
        density = density.select(energy_coordinates)

    dn, dk = normal - ref_normal, pairing - ref_pairing
    # Explicit Wick polynomial in the density differences, in MeanFi's pairing convention.
    expected_energy = (
        np.trace(h0 @ normal).real
        + interaction
        * (dn[0, 0].real * dn[1, 1].real - abs(dn[0, 1]) ** 2 - abs(dk[0, 1]) ** 2)
    ) / 2
    normal_h = h0 + interaction * np.array(
        [[dn[1, 1], -dn[0, 1]], [-dn[1, 0], dn[0, 0]]]
    )
    gap = -interaction * dk
    expected_h = np.block([[normal_h, gap], [gap.conj().T, -normal_h.T]])

    with monkeypatch.context() as patch:
        if use_sparse:

            def forbid_dense(*args, **kwargs):
                raise AssertionError(
                    "reference mapping and energy must not densify sparse blocks"
                )

            patch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
        model = replace(bare, reference=reference)
        h = model.hamiltonian_from_density(density)[()]
        error = abs(mf.trial_internal_energy(model, density) - expected_energy)
        assert error < 1e-14, f"BdG reference energy error: {error}"
        assert mf.trial_free_energy(model, density) == pytest.approx(
            expected_energy - model.kT * entropy, abs=1e-14
        )

    np.testing.assert_allclose(
        h.toarray() if use_sparse else h, expected_h, atol=1e-14, rtol=0
    )
    # Density variations have no identity term in their hole block.
    dnormal = np.array([[0.02, 0.05j], [-0.05j, -0.03]])
    dpairing = np.array([[0.0, 0.017 + 0.03j], [-0.017 - 0.03j, 0.0]])
    direction = np.block([[dnormal, dpairing], [dpairing.conj().T, -dnormal.T]])
    step = 1e-5
    derivative = (
        mf.trial_internal_energy(model, tb(raw + step * direction))
        - mf.trial_internal_energy(model, tb(raw - step * direction))
    ) / (2 * step)
    expected_derivative = np.trace(expected_h @ direction).real / 4
    error = abs(derivative - expected_derivative)
    assert error < 1e-10, f"BdG reference energy derivative error: {error}"

    # Rotate state and reference together; a fixed paired reference sets a phase.
    phase = np.exp(0.74j)
    rotated_reference = (
        reference
        if normal_reference
        else density_result_from_tb({(): covariance(ref_normal, phase * ref_pairing)})
    )
    rotated_model = replace(model, reference=rotated_reference)
    assert mf.trial_internal_energy(
        rotated_model, tb(covariance(normal, phase * pairing))
    ) == pytest.approx(expected_energy, abs=1e-14)


@pytest.mark.parametrize("manual_reference", [False, True])
@pytest.mark.parametrize("normal_reference", [False, True])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_reference_bdg_ediis_matches_scalar_gap_equation(
    normal_reference, use_sparse, manual_reference
):
    if use_sparse:
        pytest.importorskip("mumps")
    interaction, temperature, onsite = 1.6, 0.15, 0.23
    phase = np.exp(0.37j)
    ref_amplitude = 0.0 if normal_reference else 0.05
    normal = 0.5 * np.eye(2)
    ref_pairing = np.array(
        [[0.0, -ref_amplitude * phase], [ref_amplitude * phase, 0.0]]
    )
    reference = density_result_from_tb(
        {(): normal if normal_reference else covariance(normal, ref_pairing)}
    )
    if manual_reference:
        reference = reference.to_tb(sparse=use_sparse)
    h0 = {(): onsite * np.eye(2)}
    hint = {(): np.array([[0.0, interaction], [interaction, 0.0]])}
    if use_sparse:
        h0 = {key: sparse.csr_matrix(value) for key, value in h0.items()}
        hint = {key: sparse.csr_matrix(value) for key, value in hint.items()}
    model = mf.Model(
        h0, hint, filling=1.0, kT=temperature, superconducting=True, reference=reference
    )
    gap_guess = np.array([[0.0, -0.4 * phase], [0.4 * phase, 0.0]])
    guess = {
        (): np.block(
            [[np.zeros((2, 2)), gap_guess], [gap_guess.conj().T, np.zeros((2, 2))]]
        )
    }
    grid = mf.UniformGrid(
        matrix_function=mf.RationalFOE() if use_sparse else mf.DirectDiagonalization(),
    )
    result = mf.solver(
        model, guess, integration=grid, tol=1e-9, compute_free_energy=True
    )

    # Independent positive gap: x = V * (tanh(x / (2 kT)) / 2 + reference_pairing).
    gap = brentq(
        lambda x: x
        - interaction * (0.5 * np.tanh(x / (2 * temperature)) + ref_amplitude),
        1e-5,
        interaction,
        xtol=1e-14,
    )
    expected_energy = onsite / 2 - gap**2 / (2 * interaction)
    occupation = expit(-gap / temperature)
    expected_entropy = entr(occupation) + entr(1 - occupation)
    assert result.converged and len(result.history) > 1
    assert abs(result.mu - onsite) < 2e-8
    assert abs(result.mean_field[()][0, 3] + gap * phase) < 2e-8
    assert abs(result.internal_energy - expected_energy) < 2e-8
    entropy_error = result.errors.entropy or 0.0
    assert abs(result.entropy - expected_entropy) < entropy_error + 2e-8
    assert (
        abs(result.free_energy - (expected_energy - temperature * expected_entropy))
        < temperature * entropy_error + 2e-8
    )
    assert mf.trial_internal_energy(model, result.density) == pytest.approx(
        result.internal_energy, abs=2e-8
    )


@pytest.mark.parametrize("normal_reference", [False, True])
def test_bdg_reference_rejects_missing_required_entries(normal_reference):
    bare = mf.Model(
        {(): np.eye(2)},
        {(): np.array([[0.0, 1.0], [1.0, 0.0]])},
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )
    ref_model = replace(bare, superconducting=not normal_reference)
    required = ref_model.required_coordinates
    coordinates = mf.DensityCoordinates.from_entries(
        size=required.size, keys=[()], entries=required.entries[:-1]
    )
    reference = mf.DensityResult(
        entries=_DensityEntries(coordinates, np.zeros(coordinates.value_count)),
        mu=0.0,
        filling=1.0,
        errors=mf.ErrorValues(),
    )
    with pytest.raises(ValueError, match="missing .* required coordinate"):
        replace(bare, reference=reference)


@pytest.mark.parametrize("with_symmetry", [False, True])
def test_normal_reference_matches_bdg_reference_on_nonlocal_model(with_symmetry):
    symmetries = (
        (mf.SpatialSymmetry(np.array([[-1]]), {(0,): np.eye(2)}),)
        if with_symmetry
        else ()
    )
    normal = mf.Model(
        {
            (0,): np.array([[0.1, 0.03], [0.03, -0.1]]),
            (1,): -np.eye(2),
            (-1,): -np.eye(2),
        },
        {
            (0,): np.array([[0.0, 0.5], [0.5, 0.0]]),
            (1,): 0.2 * np.ones((2, 2)),
            (-1,): 0.2 * np.ones((2, 2)),
        },
        filling=0.8,
        kT=0.2,
        spatial_symmetries=symmetries,
    )
    grid = mf.UniformGrid(nk=64)
    normal_reference = mf.density_matrix(normal, integration=grid, tol=1e-9)
    bdg = replace(normal, superconducting=True)
    bdg_reference = mf.density_matrix(bdg, integration=grid, tol=1e-9)
    from_normal = replace(bdg, reference=normal_reference)
    from_bdg = replace(bdg, reference=bdg_reference)
    np.testing.assert_allclose(
        from_normal._reference_state.values,
        from_bdg._reference_state.values,
        atol=1e-9,
        rtol=0,
    )
    actual = from_normal.hamiltonian_from_density(bdg_reference)
    expected = bdg.hamiltonian_from_meanfield()
    for key in actual:
        np.testing.assert_allclose(
            actual[key], expected.get(key, np.zeros((4, 4))), atol=1e-9, rtol=0
        )
