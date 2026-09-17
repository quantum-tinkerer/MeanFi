"""Finite-range normal and anomalous contractions against a six-mode Fock space."""

from dataclasses import replace

import numpy as np
import pytest

import meanfi as mf
from meanfi.meanfield import correction_expectation
from meanfi.tests.fixtures.fermions import annihilators
from meanfi.tb.bdg import validate_bdg_tb
from meanfi.tb.validate import validate_hermiticity

pytestmark = pytest.mark.physics


def _ring_gaussian_state(superconducting, seed):
    """Thermal Gaussian state of three cells, with two orbitals per cell."""
    rng = np.random.default_rng(seed)
    cells, orbitals = 3, 2
    c = annihilators(cells * orbitals)
    onsite = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    onsite = 0.2 * (onsite + onsite.conj().T)
    hop = 0.2 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    gap = 0.3 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    local_gap = 0.2j * np.array([[0, 1], [-1, 0]])
    quadratic = np.zeros((64, 64), complex)
    for x in range(cells):
        for i in range(orbitals):
            for j in range(orbitals):
                quadratic += onsite[i, j] * c[2 * x + i].conj().T @ c[2 * x + j]
                bond = hop[i, j] * c[2 * x + i].conj().T @ c[2 * ((x + 1) % cells) + j]
                quadratic += bond + bond.conj().T
                if superconducting:
                    pair = (
                        gap[i, j]
                        * c[2 * x + i].conj().T
                        @ c[2 * ((x + 1) % cells) + j].conj().T
                    )
                    pair += (
                        0.5
                        * local_gap[i, j]
                        * c[2 * x + i].conj().T
                        @ c[2 * x + j].conj().T
                    )
                    quadratic += pair + pair.conj().T
    energies, vectors = np.linalg.eigh(quadratic)
    weights = np.exp(-(energies - energies.min()) / 0.4)
    state = (vectors * (weights / weights.sum())) @ vectors.conj().T
    rho = np.array(
        [[np.trace(state @ right.conj().T @ left) for right in c] for left in c]
    )
    kappa = np.array([[np.trace(state @ right @ left) for right in c] for left in c])
    normal, anomalous = {}, {}
    for r in (0, 1, -1):
        normal[(r,)] = rho[:2, 2 * (r % cells) : 2 * (r % cells) + 2]
        anomalous[(r,)] = kappa[:2, 2 * (r % cells) : 2 * (r % cells) + 2]
    density = (
        normal
        if not superconducting
        else {
            (r,): np.block(
                [
                    [normal[(r,)], anomalous[(r,)]],
                    [
                        anomalous[(-r,)].conj().T,
                        (np.eye(2) if r == 0 else 0) - normal[(-r,)].T,
                    ],
                ]
            )
            for r in (0, 1, -1)
        }
    )
    return state, c, density


def _fock_interaction(terms, c):
    def one_body(matrix, cell):
        return sum(
            matrix[i, j] * c[2 * cell + i].conj().T @ c[2 * cell + j]
            for i in range(2)
            for j in range(2)
        )

    result = np.zeros((64, 64), complex)
    for term in terms:
        displacement = 0 if term.displacement is None else term.displacement[0]
        for x in range(3):
            y = (x + displacement) % 3
            product = one_body(term.A, x) @ one_body(term.B, y)
            if x == y:
                product -= one_body(term.A @ term.B, x)
            result += term.coefficient * product
    return result


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("displacement", [None, (1,), (-1,)])
def test_wick_energy_and_gradient_against_exact_ring(superconducting, displacement):
    rng = np.random.default_rng(73)
    a, b = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    a, b = a + a.conj().T, b + b.conj().T
    interaction = mf.BilinearInteraction(
        [mf.BilinearTerm(0.17, a, b, displacement=displacement)]
    )
    state, c, density = _ring_gaussian_state(superconducting, 9)
    model = mf.Model(
        {(0,): np.zeros((2, 2))}, interaction, 1, superconducting=superconducting
    )
    expected = np.trace(state @ _fock_interaction(interaction.terms, c)).real / 6
    error = abs(mf.evaluate_internal_energy(model, density) - expected)
    assert error < 2e-12, f"Exact Fock-space energy error per orbital: {error}"
    correction = model.mean_field(density)
    if superconducting:
        validate_bdg_tb(correction, ndof=2, ndim=1)
    else:
        validate_hermiticity(correction)
    _, _, other = _ring_gaussian_state(superconducting, 23)
    direction = {key: other[key] - density[key] for key in density}
    step = 1e-5
    plus = {key: density[key] + step * direction[key] for key in density}
    minus = {key: density[key] - step * direction[key] for key in density}
    derivative = (
        mf.evaluate_internal_energy(model, plus)
        - mf.evaluate_internal_energy(model, minus)
    ) / (2 * step)
    predicted = correction_expectation(
        direction, correction, electron_ndof=2 if superconducting else None
    )
    assert derivative == pytest.approx(predicted, abs=3e-9)
    # Exchanging the two bilinears reverses R without changing the operator.
    reverse = None if displacement is None else (-displacement[0],)
    reversed_model = replace(
        model,
        h_int=mf.BilinearInteraction(
            [mf.BilinearTerm(0.17, b, a, displacement=reverse)]
        ),
    )
    for key, block in correction.items():
        np.testing.assert_allclose(
            reversed_model.mean_field(density)[key], block, atol=2e-14
        )


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("displacement", [(0,), (1,)])
def test_intercell_hubbard_equivalence_and_reference(superconducting, displacement):
    _, _, density = _ring_gaussian_state(superconducting, 12)
    _, _, reference = _ring_gaussian_state(superconducting, 15)
    coupling = -0.8
    a, b = np.diag([1, 0]), np.diag([0, 1])
    general = mf.BilinearInteraction(
        [mf.BilinearTerm(coupling, a, b, displacement=displacement)]
    )
    opposite = (-displacement[0],)
    legacy = {key: np.zeros((2, 2)) for key in {(0,), displacement, opposite}}
    legacy[displacement][0, 1] += coupling
    legacy[opposite][1, 0] += coupling
    kwargs = dict(
        h_0={(0,): np.diag([-0.2, 0.2])},
        filling=1,
        superconducting=superconducting,
        reference=reference,
    )
    first, second = mf.Model(h_int=general, **kwargs), mf.Model(h_int=legacy, **kwargs)
    for key, block in first.mean_field(density).items():
        np.testing.assert_allclose(block, second.mean_field(density)[key], atol=2e-14)
    assert mf.evaluate_internal_energy(first, density) == pytest.approx(
        mf.evaluate_internal_energy(second, density), abs=2e-14
    )


def test_displacement_validation_and_ownership():
    displacement = [1, -2]
    term = mf.BilinearTerm(1, np.eye(2), np.eye(2), displacement=displacement)
    displacement[0] = 10
    assert term.displacement == (1, -2)
    with pytest.raises(ValueError, match="dimension"):
        mf.Model({(0,): np.eye(2)}, mf.BilinearInteraction([term]), 1)
    for invalid in ([True], [0.5], ["x"]):
        with pytest.raises(ValueError, match="integers"):
            mf.BilinearTerm(1, np.eye(2), np.eye(2), displacement=invalid)
