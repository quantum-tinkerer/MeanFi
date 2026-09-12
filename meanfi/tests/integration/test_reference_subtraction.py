import numpy as np
import pytest

from meanfi import (
    PeriodicGrid,
    DensityCoordinates,
    DensityResult,
    ErrorValues,
    LinearMixing,
    Model,
    density_matrix,
    density_matrix_at_mu,
    solver,
)
from meanfi.meanfield import meanfield
from meanfi.space.state import ActiveDensityState


pytestmark = pytest.mark.integration


def test_selected_density_is_an_efficient_reference_without_zero_filling():
    h_0 = {(): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)}
    h_int = {(): np.array([[0.5, 1.2], [1.2, 0.8]], dtype=complex)}
    reference = density_matrix(
        h_0,
        filling=1.0,
        kT=0.2,
        interaction=h_int,
        integration=PeriodicGrid(density_matrix_tol=1e-10),
        filling_tol=1e-10,
    )
    model = Model(h_0, h_int, filling=1.0, kT=0.2, reference=reference)

    assert reference.is_complete is False
    assert reference.coordinates.entries == model.scf_space.required_coordinates.entries
    assert reference.coordinates.value_count < 2**2
    assert model.reference is reference
    with pytest.raises(ValueError, match="selected density coordinates"):
        reference.to_matrix()
    np.testing.assert_allclose(
        model.hamiltonian_from_rho(reference)[()],
        h_0[()],
        atol=1e-12,
    )


def test_model_rejects_selected_reference_missing_an_interaction_coordinate():
    h_0 = {(): np.zeros((2, 2), dtype=complex)}
    h_int = {(): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)}
    coordinates = DensityCoordinates.from_entries(
        size=2,
        keys=[()],
        entries=(((), 0, 0),),
        allow_empty=False,
    )
    assert coordinates is not None
    reference = DensityResult(
        coordinates=coordinates,
        values=np.array([0.5]),
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )

    with pytest.raises(ValueError, match="missing .* required coordinate"):
        Model(h_0, h_int, filling=1.0, reference=reference)


def test_reference_density_subtracts_full_mean_field_correction():
    h_0 = {(): np.array([[0.3, 0.7], [0.7, -0.2]], dtype=complex)}
    h_int = {(): np.array([[0.5, 1.2], [1.2, 0.8]], dtype=complex)}
    rho_ref = {
        (): np.array(
            [[0.6, 0.15 + 0.08j], [0.15 - 0.08j, 0.4]],
            dtype=complex,
        )
    }
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.2,
        reference_density_matrix=rho_ref,
    )

    unsubtracted_correction = meanfield(rho_ref, h_int)[()]
    hamiltonian = model.hamiltonian_from_rho(rho_ref)

    assert abs(unsubtracted_correction[0, 1]) > 1e-12
    np.testing.assert_allclose(hamiltonian[()], h_0[()], atol=1e-12)


def test_solver_reference_density_fixed_point_has_zero_interaction_correction():
    h_0 = {(): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)}
    h_int = {(): np.array([[0.4, 1.1], [1.1, 0.6]], dtype=complex)}
    integration = PeriodicGrid(density_matrix_tol=1e-10)
    rho_ref = density_matrix(
        h_0,
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=integration,
        filling_tol=1e-10,
    ).density_matrix
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.2,
        reference_density_matrix=rho_ref,
    )

    result = solver(
        model,
        {(): np.zeros((2, 2), dtype=complex)},
        integration=integration,
        scf=LinearMixing(max_iterations=3, alpha=1.0),
        scf_tol=1e-8,
        filling_tol=1e-10,
    )
    interaction_correction = result.mean_field[()]
    final_density = density_matrix_at_mu(
        model.hamiltonian_from_meanfield(result.mean_field),
        result.mu,
        kT=model.kT,
        keys=[()],
        integration=integration,
    )

    np.testing.assert_allclose(
        final_density.density_matrix[()],
        rho_ref[()],
        atol=1e-8,
    )
    np.testing.assert_allclose(interaction_correction, np.zeros((2, 2)), atol=1e-8)


def test_reference_is_a_private_read_only_active_density_state():
    h_0 = {(): np.zeros((2, 2), dtype=complex)}
    h_int = {(): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)}
    rho_ref = {
        (): np.array(
            [[0.6, 0.1 + 0.2j], [0.1 - 0.2j, 0.4]],
            dtype=complex,
        )
    }
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        reference_density_matrix=rho_ref,
    )

    assert not hasattr(model, "reference_density_matrix")
    reference_state = model._reference_state
    assert isinstance(reference_state, ActiveDensityState)
    assert reference_state.space is model.scf_space
    assert reference_state.values.flags.writeable is False
    expected = model.scf_space.params_from_meanfield_input(rho_ref)
    np.testing.assert_allclose(reference_state.values, expected)

    rho_ref[()][0, 0] = 0.0
    np.testing.assert_allclose(reference_state.values, expected)
    with pytest.raises(ValueError):
        reference_state.values[0] = 0.0
