from dataclasses import fields

import numpy as np
import pytest

from meanfi import (
    AdaptiveQuadrature,
    DensityResult,
    ErrorValues,
    SCFIteration,
    SCFResult,
    density_matrix,
)
from meanfi.density.internal import DensitySlice
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates


pytestmark = pytest.mark.integration


def test_density_selection_modes_preserve_the_explicit_layout():
    hamiltonian = {(): np.diag([-0.5, 0.5]).astype(complex)}
    integration = AdaptiveQuadrature(density_matrix_tol=1e-10)
    coordinates = DensityCoordinates.from_entries(
        size=2,
        keys=[()],
        entries=(((), 0, 0),),
        allow_empty=False,
    )
    assert coordinates is not None

    complete = density_matrix(
        hamiltonian,
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=integration,
        filling_tol=1e-10,
    )
    selected = density_matrix(
        hamiltonian,
        filling=1.0,
        kT=0.2,
        coordinates=coordinates,
        integration=integration,
        filling_tol=1e-10,
    )

    assert complete.is_complete is True
    assert selected.coordinates.entries == coordinates.entries
    assert selected.is_complete is False
    np.testing.assert_allclose(selected.values, complete.values_for(coordinates))


def test_density_requires_exactly_one_selection_mode():
    hamiltonian = {(): np.eye(1, dtype=complex)}

    with pytest.raises(ValueError, match="exactly one"):
        density_matrix(hamiltonian, filling=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        density_matrix(
            hamiltonian,
            filling=0.5,
            keys=[()],
            interaction={(): np.zeros((1, 1), dtype=complex)},
        )


def test_density_rejects_coordinates_for_a_different_matrix_size():
    coordinates = full_density_coordinates([()], size=2)

    with pytest.raises(ValueError, match="coordinate matrix size"):
        density_matrix(
            {(): np.ones((1, 1), dtype=complex)},
            filling=0.5,
            coordinates=coordinates,
        )


def test_public_result_objects_have_only_physical_values_and_achieved_errors():
    assert [field.name for field in fields(DensityResult)] == [
        "coordinates",
        "values",
        "mu",
        "filling",
        "errors",
    ]
    assert [field.name for field in fields(SCFIteration)] == [
        "step",
        "mu",
        "filling",
        "total_energy",
        "errors",
    ]
    assert [field.name for field in fields(SCFResult)] == [
        "density",
        "mean_field",
        "total_energy",
        "errors",
        "history",
        "converged",
    ]

    coordinates = full_density_coordinates([(0,)], size=2)
    density = DensityResult(
        coordinates=coordinates,
        values=np.zeros(coordinates.value_count),
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )
    result = SCFResult(
        density=density,
        mean_field={(0,): np.zeros((2, 2))},
        total_energy=None,
        errors=ErrorValues(scf_residual=0.0),
        history=(),
        converged=True,
    )
    assert not hasattr(result, "density_matrix")
    assert not hasattr(result, "effective_hamiltonian")
    assert not hasattr(result, "integration")
    assert not hasattr(result, "tolerances")
    assert result.mu == density.mu
    assert result.filling == density.filling


def test_selected_density_slice_cannot_be_exposed_as_a_full_matrix():
    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,)],
        pairs_by_key={(0,): (np.array([0]), np.array([1]))},
        allow_empty=False,
    )
    assert coordinates is not None
    density = DensitySlice(coordinates, np.array([0.25 + 0.5j]))

    assert coordinates.is_full is False
    with pytest.raises(ValueError, match="selected density entries"):
        density.to_full_tb()

    result = DensityResult(
        coordinates=coordinates,
        values=density.values,
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )
    assert result.is_complete is False
    with pytest.raises(ValueError, match="selected density coordinates"):
        result.to_matrix()


def test_density_layout_and_values_are_read_only():
    coordinates = full_density_coordinates([(0,)], size=2)
    density = DensitySlice(
        coordinates,
        np.arange(4, dtype=float).astype(complex),
        np.zeros(4),
    )

    with pytest.raises(ValueError):
        density.values[0] = 3.0
    with pytest.raises(ValueError):
        density.errors[0] = 1.0
    with pytest.raises(ValueError):
        coordinates.rows_by_key[0][0] = 1

    matrix = density.to_full_tb()[(0,)]
    np.testing.assert_array_equal(matrix, np.arange(4).reshape(2, 2))

    result = DensityResult(
        coordinates=coordinates,
        values=density.values,
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )
    with pytest.raises(ValueError):
        result.values[0] = 3.0
    np.testing.assert_array_equal(
        result.density_matrix[(0,)],
        np.arange(4).reshape(2, 2),
    )


def test_active_density_state_is_read_only_and_tied_to_one_active_space():
    class Space:
        num_params = 2

    space = Space()
    state = ActiveDensityState(space, np.array([0.1, -0.2]))
    reference = ActiveDensityState(space, np.array([0.05, 0.1]))
    difference = state.relative_to(reference)

    with pytest.raises(ValueError):
        state.values[0] = 0.0
    require_same_space(state, space)
    np.testing.assert_allclose(difference.values, np.array([0.05, -0.3]))
    assert difference.space is space
    with pytest.raises(ValueError, match="different active space"):
        require_same_space(state, Space())
    with pytest.raises(ValueError, match="different active space"):
        state.relative_to(ActiveDensityState(Space(), np.zeros(2)))
