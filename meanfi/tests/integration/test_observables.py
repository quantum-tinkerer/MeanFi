import numpy as np
import pytest

from meanfi import Model, add_tb, expectation_value, total_energy
from meanfi.meanfield import (
    bdg_correction_from_density,
    extract_anomalous_density,
    extract_electron_density,
    meanfield,
)
from meanfi.tests.fixtures.models import bipartite_hubbard_2d


pytestmark = pytest.mark.integration


def test_total_energy_half_counts_normal_mean_field_interaction():
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

    assert total_energy(model, density) == pytest.approx(np.real(expected))
    assert naive - total_energy(model, density) == pytest.approx(
        0.5 * interaction_energy
    )


def test_total_energy_uses_reference_subtracted_interaction_functional():
    h_0 = {(): np.diag([0.2, -0.3]).astype(complex)}
    h_int = {(): np.array([[0.0, 1.7], [1.7, 0.0]], dtype=complex)}
    reference = {(): np.diag([0.6, 0.4]).astype(complex)}
    density = {(): np.diag([0.25, 0.75]).astype(complex)}
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        reference_density_matrix=reference,
    )
    difference = {(): density[()] - reference[()]}
    correction = meanfield(difference, h_int)
    expected = expectation_value(density, h_0)
    expected += 0.5 * expectation_value(difference, correction)

    assert total_energy(model, density) == pytest.approx(float(np.real(expected)))


def test_total_energy_rejects_missing_one_body_density_keys():
    model = Model(
        {
            (0,): np.zeros((1, 1), dtype=complex),
            (1,): np.ones((1, 1), dtype=complex),
            (-1,): np.ones((1, 1), dtype=complex),
        },
        {(0,): np.zeros((1, 1), dtype=complex)},
        filling=0.5,
    )

    with pytest.raises(ValueError, match="missing keys required for total energy"):
        total_energy(model, {(0,): np.array([[0.5]], dtype=complex)})


def test_total_energy_gradient_matches_hubbard_mean_field_hamiltonian():
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
        total_energy(model, shifted(epsilon)) - total_energy(model, shifted(-epsilon))
    ) / (2.0 * epsilon)
    rhs = np.real(expectation_value(direction, model.hamiltonian_from_rho(rho)))

    assert total_energy(model, rho) == pytest.approx(slater_energy())
    assert derivative == pytest.approx(rhs, rel=1e-8, abs=1e-8)


def test_total_energy_matches_bdg_block_formula():
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

    correction = bdg_correction_from_density(density, model)
    electron_density = extract_electron_density(density, model)
    anomalous_density = extract_anomalous_density(density, model)
    normal_correction = {
        key: matrix[: model._ndof, : model._ndof] for key, matrix in correction.items()
    }
    pairing_correction = {
        key: matrix[: model._ndof, model._ndof :] for key, matrix in correction.items()
    }
    expected = expectation_value(electron_density, model.h_0)
    expected += 0.5 * expectation_value(electron_density, normal_correction)
    expected += 0.5 * expectation_value(anomalous_density, pairing_correction)

    assert total_energy(model, density) == pytest.approx(np.real(expected))


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

    correction = bdg_correction_from_density(density, model)

    assert correction[(1,)][0, 1] == pytest.approx(-0.20000002)
    assert correction[(-1,)][0, 1] == pytest.approx(0.20000002)
