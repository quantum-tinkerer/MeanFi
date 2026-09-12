import numpy as np
import pytest

from meanfi import (
    AndersonMixing,
    LinearMixing,
    Model,
    PeriodicGrid,
    solver,
)
from meanfi.density.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def test_superconducting_model_uses_electron_first_bdg_embedding():
    model = Model(
        {(): np.array([[2.0]], dtype=complex)},
        {(): np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )

    hamiltonian = model.bdg_hamiltonian_from_meanfield(
        {(): np.zeros((2, 2), dtype=complex)}
    )

    assert np.allclose(hamiltonian[()], np.diag([2.0, -2.0]))


def test_bdg_solver_validates_guess_shape_before_running_density():
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )

    with pytest.raises(ValueError, match="2\\*ndof"):
        solver(
            model,
            {(0,): np.zeros((2, 2), dtype=complex)},
            integration=PeriodicGrid(),
        )


def test_bdg_solver_rejects_guess_without_opposite_key():
    model = Model(
        {
            (0,): np.array([[0.0]], dtype=complex),
            (1,): np.array([[0.0]], dtype=complex),
            (-1,): np.array([[0.0]], dtype=complex),
        },
        {(0,): np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )

    with pytest.raises(ValueError, match="opposite keys"):
        solver(
            model,
            {(1,): np.zeros((2, 2), dtype=complex)},
            integration=PeriodicGrid(),
        )


def test_bdg_solver_rejects_guess_with_invalid_block_structure():
    model = Model(
        {(0,): np.array([[0.0]], dtype=complex)},
        {(0,): np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    guess = {
        (0,): np.array([[1.0, 0.2], [0.2, 1.0]], dtype=complex),
    }

    with pytest.raises(ValueError, match="lower-right block"):
        solver(
            model,
            guess,
            integration=PeriodicGrid(),
        )


def test_bdg_solver_supports_anderson_mixing():
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )

    result = solver(
        model,
        {(0,): np.zeros((4, 4), dtype=complex)},
        integration=PeriodicGrid(),
        scf=AndersonMixing(M=0, max_iterations=4),
    )

    assert result.history
    assert result.errors.scf_residual is not None


def test_zero_temperature_bdg_requires_explicit_periodic_grid_default_override():
    model = Model(
        {(0,): np.array([[0.0]], dtype=complex)},
        {(0,): np.array([[0.0]], dtype=complex)},
        filling=0.5,
        superconducting=True,
    )

    with pytest.raises(NotImplementedError, match="PeriodicGrid"):
        solver(
            model,
            {(0,): np.zeros((2, 2), dtype=complex)},
        )


def test_zero_temperature_bdg_supports_explicit_periodic_grid():
    model = Model(
        {(0,): np.array([[0.0]], dtype=complex)},
        {(0,): np.array([[0.0]], dtype=complex)},
        filling=0.5,
        superconducting=True,
    )

    result = solver(
        model,
        {(0,): np.zeros((2, 2), dtype=complex)},
        integration=PeriodicGrid(nk=1),
        scf=LinearMixing(max_iterations=2),
        scf_tol=1e-6,
    )

    assert np.isfinite(result.mu)


def test_bdg_solver_warns_when_guess_is_projected_to_structural_selection():
    model = Model(
        {(0,): np.zeros((2, 2), dtype=complex)},
        {(0,): np.zeros((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )
    anomalous = np.array([[0.0, 0.3], [-0.3, 0.0]], dtype=complex)
    guess = {
        (0,): np.block(
            [
                [np.zeros((2, 2), dtype=complex), anomalous],
                [anomalous.conj().T, np.zeros((2, 2), dtype=complex)],
            ]
        )
    }

    with pytest.warns(UserWarning, match="projected away"):
        result = solver(
            model,
            guess,
            integration=PeriodicGrid(density_matrix_tol=1e-2),
            scf=LinearMixing(max_iterations=1),
            scf_tol=1e-8,
        )

    assert result.errors.scf_residual is not None


def test_model_random_meanfield_generates_valid_bdg_guess():
    model = Model(
        spinful_chain(),
        {(0,): np.ones((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )

    first = model.random_meanfield(rng=123, scale=0.1)
    second = model.random_meanfield(rng=123, scale=0.1)
    zero = model.random_meanfield(rng=123, scale=0.0)

    assert set(first) == {(0,)}
    for key in first:
        np.testing.assert_allclose(first[key], second[key])
        np.testing.assert_allclose(zero[key], np.zeros_like(zero[key]))
    model.bdg_hamiltonian_from_meanfield(first)
