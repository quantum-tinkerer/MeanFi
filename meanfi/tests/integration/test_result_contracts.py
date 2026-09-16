from meanfi import default_solver_tolerances
from meanfi.results import _DensityEntries
from dataclasses import fields, replace

import numpy as np
import pytest

from meanfi import (
    UniformGrid,
    DensityResult,
    ErrorValues,
    SCFIteration,
    SCFResult,
    density_matrix,
    density_matrix_at_mu,
)
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates


pytestmark = pytest.mark.integration


def test_density_selection_modes_preserve_the_explicit_layout():
    hamiltonian = {(): np.diag([-0.5, 0.5]).astype(complex)}
    integration = UniformGrid()
    coordinates = DensityCoordinates.from_entries(
        size=2,
        keys=[()],
        entries=(((), 0, 0),),
    )

    complete = density_matrix(
        hamiltonian,
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=integration,
        tol=replace(
            default_solver_tolerances(1e-3),
            density_matrix_integration=1e-10,
            charge_integration=1e-10,
            filling_residual=1e-10,
        ),
    )
    selected = density_matrix(
        hamiltonian,
        filling=1.0,
        kT=0.2,
        coordinates=coordinates,
        integration=integration,
        tol=replace(
            default_solver_tolerances(1e-3),
            density_matrix_integration=1e-10,
            charge_integration=1e-10,
            filling_residual=1e-10,
        ),
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


def test_public_result_objects_report_physics_errors_and_density_statistics():
    assert [field.name for field in fields(SCFIteration)] == [
        "step",
        "mu",
        "filling",
        "internal_energy",
        "errors",
    ]
    assert [field.name for field in fields(SCFResult)] == [
        "density",
        "mean_field",
        "history",
        "converged",
    ]

    coordinates = full_density_coordinates([(0,)], size=2)
    density = DensityResult(
        entries=_DensityEntries(coordinates, np.zeros(coordinates.value_count)),
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )
    result = SCFResult(
        density=replace(density, errors=ErrorValues(scf_residual=0.0)),
        mean_field={(0,): np.zeros((2, 2))},
        history=(),
        converged=True,
    )
    assert not hasattr(result, "density_matrix")
    assert not hasattr(result, "effective_hamiltonian")
    assert not hasattr(result, "integration")
    assert not hasattr(result, "tolerances")
    assert result.mu == density.mu
    assert result.filling == density.filling


def test_selected_density_entries_cannot_be_exposed_as_a_full_matrix():
    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,)],
        pairs_by_key={(0,): (np.array([0]), np.array([1]))},
    )
    density = _DensityEntries(coordinates, np.array([0.25 + 0.5j]))

    assert coordinates.is_full is False
    with pytest.raises(ValueError, match="selected density coordinates"):
        density.to_tb()

    result = DensityResult(
        entries=density,
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )
    assert result.is_complete is False
    with pytest.raises(ValueError, match="selected density coordinates"):
        result.to_tb()


def test_density_layout_and_values_are_read_only():
    coordinates = full_density_coordinates([(0,)], size=2)
    density = _DensityEntries(
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

    matrix = density.to_tb()[(0,)]
    np.testing.assert_array_equal(matrix, np.arange(4).reshape(2, 2))

    result = DensityResult(
        entries=density,
        mu=0.0,
        filling=1.0,
        errors=ErrorValues(),
    )
    with pytest.raises(ValueError):
        result.values[0] = 3.0
    np.testing.assert_array_equal(
        result.to_tb()[(0,)],
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


@pytest.mark.parametrize(
    "density_function, target",
    [
        (density_matrix, {"filling": 1.0}),
        (density_matrix_at_mu, {"mu": 0.0}),
    ],
)
def test_both_density_apis_support_all_selection_modes(density_function, target):
    h = {(): np.diag([-0.5, 0.5]).astype(complex)}
    interaction = {(): np.diag([1.0, 0.0])}
    full = density_function(h, kT=0.2, keys=[()], **target)
    selected = density_function(h, kT=0.2, interaction=interaction, **target)
    exact = density_function(h, kT=0.2, coordinates=selected.coordinates, **target)
    np.testing.assert_allclose(selected.values, full.values_for(selected.coordinates))
    np.testing.assert_allclose(exact.values, selected.values)
    assert exact.coordinates is selected.coordinates
    assert selected.filling == pytest.approx(full.filling)
    for selection in (
        {},
        {"keys": [()], "coordinates": full.coordinates},
        {"interaction": interaction, "coordinates": full.coordinates},
    ):
        with pytest.raises(ValueError, match="exactly one"):
            density_function(h, **target, **selection)


def test_results_share_immutable_entries_and_preserve_selected_errors():
    coordinates = full_density_coordinates([()], size=2)
    values = np.arange(4, dtype=complex)
    errors = np.arange(4, dtype=float) / 10
    entries = _DensityEntries(coordinates, values, errors)
    result = DensityResult(
        entries, mu=0.25, filling=1.5, errors=ErrorValues(), band_energy=-0.75
    )
    updated = replace(result, errors=ErrorValues(scf_residual=1e-6))
    assert updated.entries is entries
    assert updated.values is entries.values
    values[:] = -1
    errors[:] = -1
    np.testing.assert_array_equal(result.values, np.arange(4))
    np.testing.assert_allclose(result.entry_errors, np.arange(4) / 10)
    assert result.select(coordinates) is result
    selected_coordinates = DensityCoordinates.from_entries(
        size=2, keys=[()], entries=(((), 1, 0), ((), 0, 1))
    )
    selected = result.select(selected_coordinates)
    np.testing.assert_array_equal(
        selected.values, result.values_for(selected_coordinates)
    )
    np.testing.assert_allclose(selected.entry_errors, [0.1, 0.2])
    assert selected.mu == result.mu
    assert selected.filling == result.filling
    assert selected.errors is result.errors
    assert selected.statistics is result.statistics
    assert selected.band_energy == result.band_energy
    with pytest.raises(ValueError, match="missing"):
        selected.select(coordinates)
    empty = result.select(
        DensityCoordinates.from_entries(size=2, keys=[()], entries=())
    )
    assert empty.values.size == empty.entry_errors.size == 0
    assert (
        replace(result, entries=_DensityEntries(coordinates, result.values))
        .select(selected_coordinates)
        .entry_errors
        is None
    )


@pytest.mark.parametrize(
    "values, errors, message",
    [
        ([[1]], None, "one-dimensional"),
        ([], None, "density values do not match"),
        ([1], [], "density errors do not match"),
        ([1], [[0]], "one-dimensional"),
        ([1], [-1], "finite and non-negative"),
        ([1], [np.nan], "finite"),
        ([np.nan], None, "density values must be finite"),
        ([np.inf], None, "density values must be finite"),
    ],
)
def test_density_entries_reject_invalid_arrays(values, errors, message):
    coordinates = full_density_coordinates([()], size=1)
    with pytest.raises(ValueError, match=message):
        _DensityEntries(coordinates, values, errors)
