"""Wick and derivative references independent of the SCF implementation."""

from itertools import combinations

import numpy as np
import pytest

from meanfi import BilinearInteraction, BilinearTerm, Model, evaluate_internal_energy


pytestmark = pytest.mark.physics


def annihilators(size):
    operators = np.zeros((size, 2**size, 2**size), complex)
    for orbital in range(size):
        for state in range(2**size):
            if state & (1 << orbital):
                parity = (state & ((1 << orbital) - 1)).bit_count()
                operators[orbital, state ^ (1 << orbital), state] = (-1) ** parity
    return operators


@pytest.mark.parametrize("occupied", [1, 2, 3])
def test_bilinear_wick_and_gradient_against_fock_space(occupied):
    rng = np.random.default_rng(83 + occupied)
    orbitals = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    rho = orbitals[:, :occupied] @ orbitals[:, :occupied].conj().T
    state = np.zeros(16, complex)
    for indices in combinations(range(4), occupied):
        state[sum(1 << i for i in indices)] = np.linalg.det(
            orbitals[list(indices), :occupied]
        )
    c = annihilators(4)

    def second_quantized(matrix):
        return sum(
            matrix[i, j] * c[i].conj().T @ c[j] for i in range(4) for j in range(4)
        )

    terms = []
    exact = 0j
    for _ in range(5):
        a, b = rng.normal(size=(2, 4, 4)) + 1j * rng.normal(size=(2, 4, 4))
        a, b = a + a.conj().T, b + b.conj().T
        g = rng.normal()
        terms.append(BilinearTerm(g, a, b))
        normal_ordered = second_quantized(a) @ second_quantized(b) - second_quantized(
            a @ b
        )
        exact += g * np.vdot(state, normal_ordered @ state)
    model = Model({(): np.zeros((4, 4))}, BilinearInteraction(terms), occupied)

    def energy(density):
        return 4 * evaluate_internal_energy(model, {(): density})

    # Fock-space roundoff is ~1e-13; allow 2e-12 for five summed terms.
    assert abs(exact.imag) < 1e-12
    assert energy(rho) == pytest.approx(exact.real, abs=2e-12)
    direction = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    direction += direction.conj().T
    step = 1e-5
    derivative = (energy(rho + step * direction) - energy(rho - step * direction)) / (
        2 * step
    )
    assert derivative == pytest.approx(
        np.trace(direction @ model.mean_field({(): rho})[()]).real, abs=2e-8
    )


def test_hubbard_equivalence_with_reference_and_noncontiguous_orbitals():
    rng = np.random.default_rng(92)
    a, b = np.diag([1, 0, 0, 0]), np.diag([0, 0, 1, 0])
    interaction = BilinearInteraction([BilinearTerm(1.7, a, b)])
    density_density = np.zeros((4, 4))
    density_density[0, 2] = density_density[2, 0] = 1.7
    raw = rng.normal(size=(2, 4, 4)) + 1j * rng.normal(size=(2, 4, 4))
    rho, reference = (matrix + matrix.conj().T for matrix in raw)
    kwargs = dict(
        h_0={(0,): np.diag([-1, 0, 1, 2])}, filling=2, reference={(0,): reference}
    )
    general = Model(h_int=interaction, **kwargs)
    legacy = Model(h_int={(0,): density_density}, **kwargs)
    np.testing.assert_allclose(
        general.mean_field({(0,): rho})[(0,)],
        legacy.mean_field({(0,): rho})[(0,)],
        atol=1e-13,
    )
    assert evaluate_internal_energy(general, {(0,): rho}) == pytest.approx(
        evaluate_internal_energy(legacy, {(0,): rho}), abs=1e-13
    )
    assert all(
        row in (0, 2) and col in (0, 2)
        for _, row, col in general.required_coordinates.entries
    )


def test_bilinear_owns_operators_and_validates_terms():
    original = np.eye(2)
    term = BilinearTerm(1, original, original)
    original[0, 0] = 5
    np.testing.assert_array_equal(term.A, np.eye(2))
    assert not term.A.flags.writeable and not term.B.flags.writeable
    terms = [term]
    interaction = BilinearInteraction(terms)
    terms.clear()
    assert len(interaction.terms) == 1
    for invalid in (
        np.ones((2, 3)),
        np.array([[0, 1j], [1j, 0]]),
        np.diag([np.nan, 1]),
    ):
        with pytest.raises(ValueError, match="Hermitian"):
            BilinearTerm(1, invalid, np.eye(2))
    for coefficient in (np.inf, 1j, np.complex128(1 + 2j)):
        with pytest.raises(ValueError, match="finite and real"):
            BilinearTerm(coefficient, np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        BilinearTerm(1, np.eye(2), np.eye(3))
    with pytest.raises(ValueError, match="at least one"):
        BilinearInteraction([])
    with pytest.raises(ValueError, match="same matrix size"):
        BilinearInteraction([term, BilinearTerm(1, np.eye(3), np.eye(3))])
    with pytest.raises(ValueError, match="normal states"):
        Model({(): np.eye(2)}, interaction, 1, superconducting=True)
    with pytest.raises(ValueError, match="same dimension and matrix size"):
        Model({(): np.eye(3)}, interaction, 1)


def test_ediis_quadratic_identity_for_bilinear_reference_model():
    from meanfi.scf.problem import SCFProblem

    rng = np.random.default_rng(14)
    raw = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
    a, b, reference, h0, *densities = (x + x.conj().T for x in raw)
    model = Model(
        {(): h0},
        BilinearInteraction([BilinearTerm(0.13, a, b)]),
        filling=1,
        reference={(): reference},
    )
    # The curvature path needs only the physical model, not a density backend.
    problem = SCFProblem(model, density_problem=None)
    weights = np.array([0.2, 0.3, 0.5])
    params = [model._space.params_from_density({(): rho}) for rho in densities]
    energies = [evaluate_internal_energy(model, {(): rho}) for rho in densities]
    ediis_energy = weights @ energies - sum(
        weights[i] * weights[j] * problem.interaction_curvature(params[i] - params[j])
        for i in range(3)
        for j in range(i)
    )
    mixed = sum(w * rho for w, rho in zip(weights, densities, strict=True))
    assert ediis_energy == pytest.approx(
        evaluate_internal_energy(model, {(): mixed}), abs=2e-13
    )
