"""BdG contractions checked with exact operators in the two-orbital Fock space."""

import numpy as np
import pytest
from scipy import sparse

import meanfi as mf

pytestmark = pytest.mark.physics


def _annihilation_operators():
    lower = np.array([[0, 1], [0, 0]])
    # Basis |00>, |10>, |01>, |11>; the parity factor enforces anticommutation.
    return np.array([np.kron(np.eye(2), lower), np.kron(lower, np.diag([1, -1]))])


def _covariance(state, annihilation):
    nambu = [*annihilation, *(c.conj().T for c in annihilation)]
    return np.array(
        [[np.trace(state @ right.conj().T @ left) for right in nambu] for left in nambu]
    )


@pytest.mark.parametrize("coupling", [-2.0, 2.0])
@pytest.mark.parametrize("occupation", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("phase", [1.0, np.exp(0.37j)])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_pairing_energy_matches_exact_fock_state(
    coupling, occupation, phase, use_sparse
):
    c0, c1 = annihilation = _annihilation_operators()
    state = np.array([np.sqrt(1 - occupation), 0, 0, phase * np.sqrt(occupation)])
    state = np.outer(state, state.conj())
    density = _covariance(state, annihilation)
    interaction = coupling * c0.conj().T @ c1.conj().T @ c1 @ c0
    expected = np.trace(state @ interaction).real / 2
    model = mf.Model(
        {(): np.zeros((2, 2))},
        {(): np.array([[0, coupling], [coupling, 0]])},
        filling=2 * occupation,
        superconducting=True,
    )
    if use_sparse:
        density = sparse.csr_matrix(density)
    error = abs(mf.evaluate_internal_energy(model, {(): density}) - expected)
    assert error < 1e-14, f"Exact paired-state energy error per orbital: {error}"


@pytest.mark.parametrize("use_sparse", [False, True])
def test_thermal_bdg_density_and_energy_match_exact_fock_space(use_sparse, request):
    if use_sparse:
        request.getfixturevalue("require_mumps")
    c0, c1 = annihilation = _annihilation_operators()
    h = np.array([[0.3, 0.07j], [-0.07j, -0.2]])
    gap = 0.4 * np.exp(0.37j)
    mu, temperature, coupling = 0.11, 0.2, -1.5
    one_body = sum(
        h[i, j] * left.conj().T @ right
        for i, left in enumerate(annihilation)
        for j, right in enumerate(annihilation)
    )
    pair = c0.conj().T @ c1.conj().T
    number = c0.conj().T @ c0 + c1.conj().T @ c1
    quadratic = one_body + gap * pair + gap.conjugate() * pair.conj().T - mu * number
    energies, vectors = np.linalg.eigh(quadratic)
    weights = np.exp(-(energies - energies.min()) / temperature)
    state = (vectors * (weights / weights.sum())) @ vectors.conj().T
    expected_density = _covariance(state, annihilation)
    interaction = coupling * c0.conj().T @ c1.conj().T @ c1 @ c0
    expected_energy = np.trace(state @ (one_body + interaction)).real / 2
    expected_entropy = (
        -np.sum(weights / weights.sum() * np.log(weights / weights.sum())) / 2
    )

    def tb(matrix):
        return {(): sparse.csr_matrix(matrix) if use_sparse else matrix}

    model = mf.Model(
        tb(h),
        tb(np.array([[0, coupling], [coupling, 0]])),
        filling=1,
        kT=temperature,
        superconducting=True,
    )
    pairing = np.array([[0, gap], [-gap, 0]])
    zero = np.zeros((2, 2))
    result = mf.density_matrix_at_mu(
        model,
        mu,
        keys=[()],
        mean_field=tb(np.block([[zero, pairing], [pairing.conj().T, zero]])),
        integration=mf.UniformGrid(),
        tol=1e-9,
        compute_free_energy=True,
    )
    density_error = np.max(abs(result.to_tb()[()] - expected_density))
    energy_error = abs(result.internal_energy - expected_energy)
    assert density_error < 1e-9, f"Exact Fock-space density error: {density_error}"
    assert energy_error < 1e-9, (
        f"Exact Fock-space energy error per orbital: {energy_error}"
    )
    assert abs(result.entropy - expected_entropy) < (result.errors.entropy or 0) + 1e-9
