from dataclasses import replace
from meanfi import default_solver_tolerances
import numpy as np
import pytest

from meanfi import (
    AndersonMixing,
    LinearMixing,
    Model,
    UniformGrid,
    solver,
)
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration


def test_superconducting_model_uses_electron_first_bdg_embedding():
    model = Model(
        {(): np.array([[2.0]], dtype=complex)},
        {(): np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )

    hamiltonian = model.hamiltonian_from_meanfield(
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
            integration=UniformGrid(),
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

    with pytest.raises(ValueError, match="hermitian"):
        solver(
            model,
            {(1,): np.diag([0.1, -0.1])},
            integration=UniformGrid(),
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
            integration=UniformGrid(),
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
        integration=UniformGrid(),
        scf=AndersonMixing(history_size=0, max_iterations=4),
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

    with pytest.raises(ValueError, match="UniformGrid"):
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
        integration=UniformGrid(nk=1),
        scf=LinearMixing(max_iterations=2),
        tol=replace(default_solver_tolerances(1e-3), scf_residual=1e-6),
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
            integration=UniformGrid(),
            scf=LinearMixing(max_iterations=1),
            tol=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-2,
                charge_integration=1e-2,
                scf_residual=1e-8,
            ),
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
    model.hamiltonian_from_meanfield(first)
