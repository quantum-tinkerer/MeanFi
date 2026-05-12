import itertools as it

import numpy as np
import pytest

from meanfi.meanfield import assemble_bdg_correction
from meanfi.model import Model
from meanfi.space import (
    ActiveDensitySpace,
    DensityCoordinates,
    SpatialSymmetry,
    canonical_tb_keys,
    real_to_complex,
)
from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import compare_dicts
from meanfi.tb.transforms import ifftn_to_tb, tb_to_kfunc, tb_to_kgrid
from meanfi.tb.utils import guess_tb
from meanfi.tests.fixtures.models import qiwuzhang, spinful_chain
from meanfi.density.integrate.uniform import _selected_value_grid_to_tb


pytestmark = pytest.mark.integration


def _full_grid_tb(*, dim: int, max_order: int, matrix_size: int, seed: int):
    rng = np.random.default_rng(seed)
    keys = [range(-max_order + 1, max_order) for _ in range(dim)]
    return {
        key: rng.normal(size=(matrix_size, matrix_size))
        + 1j * rng.normal(size=(matrix_size, matrix_size))
        for key in it.product(*keys)
    }


def _coordinates() -> DensityCoordinates:
    coords = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,), (-1,)],
        pairs_by_key={
            (0,): (np.array([0]), np.array([0])),
            (-1,): (np.array([1]), np.array([0])),
        },
        allow_empty=False,
    )
    assert coords is not None
    return coords


def test_canonical_tb_keys_are_deterministic_and_explicit():
    keys = [(1, 0), (0, 0), (0, -1), (0, 1), (-1, 0)]

    assert canonical_tb_keys(keys) == [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def test_density_coordinates_value_order_and_negative_grid_key():
    coords = _coordinates()
    assert coords.value_count == 2
    np.testing.assert_array_equal(coords.all_rows, np.array([0, 1]))
    np.testing.assert_array_equal(coords.all_cols, np.array([0, 0]))
    assert list(coords.keys) == [(0,), (-1,)]
    assert coords.key_slice((-1,)) == slice(1, 2)

    values = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    selected_tb = coords.values_to_tb(values)
    assert selected_tb[(0,)][0, 0] == values[0]
    assert selected_tb[(-1,)][1, 0] == values[1]
    np.testing.assert_allclose(
        coords.values_from_tb({(0,): selected_tb[(0,)]}), [values[0], 0.0]
    )
    np.testing.assert_allclose(coords.values_from_tb(selected_tb), values)

    real_space_values = np.zeros((4, coords.value_count), dtype=complex)
    real_space_values[-1, coords.key_slice((-1,))] = values[1]
    kgrid_values = np.fft.fftn(real_space_values, axes=(0,))
    selected_from_grid = _selected_value_grid_to_tb(coords, kgrid_values, ndim=1)
    assert selected_from_grid[(-1,)][1, 0] == pytest.approx(values[1])


def test_active_density_space_required_entries_roundtrip():
    model = Model(
        spinful_chain(),
        {(0,): np.diag([1.0, 0.0]).astype(complex)},
        filling=1.0,
        kT=0.1,
    )
    space = ActiveDensitySpace.normal(model)
    values = np.array([0.25 + 0.0j])

    params = space.compress_from_entries(values)
    recovered = space.required_coordinates.values_from_tb(space.expand(params))

    assert space.required_realspace_entries() == (((0,), 0, 0),)
    np.testing.assert_allclose(recovered, values)


def test_real_to_complex_rejects_odd_length_values():
    with pytest.raises(ValueError, match="even number"):
        real_to_complex(np.array([1.0, 2.0, 3.0]))


def test_canonical_tb_keys_reject_asymmetric_key_sets():
    with pytest.raises(ValueError, match="symmetric under key inversion"):
        canonical_tb_keys([(0,), (1,)])


def test_normal_density_matrix_space_roundtrip():
    interaction = {
        (0,): np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        (1,): np.array([[0.0, 0.0], [2.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 2.0], [0.0, 0.0]], dtype=complex),
    }
    model = Model(spinful_chain(), interaction, filling=1.0, kT=0.1)
    space = ActiveDensitySpace.normal(model)
    density = {
        (0,): np.array([[0.3, 0.2j], [-0.2j, 1.7]], dtype=complex),
        (1,): np.array([[0.4, 0.5], [0.6, 0.7]], dtype=complex),
        (-1,): np.array([[0.4, 0.6], [0.5, 0.7]], dtype=complex),
    }

    projected = space.project(density)
    recovered = space.expand(space.compress(projected))
    compare_dicts(projected, recovered)


def test_bdg_space_imposes_strict_particle_hole_reduction():
    model = Model(
        spinful_chain(),
        {(0,): np.ones((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.1,
        superconducting=True,
    )
    space = ActiveDensitySpace.bdg(model)
    assert space.num_params == 6

    density = {
        (0,): np.array(
            [
                [0.2, 0.3j, 0.7, 0.1 + 0.2j],
                [-0.3j, 0.8, -0.1 - 0.2j, 0.6],
                [0.7, -0.1 + 0.2j, -0.2, -0.3j],
                [0.1 - 0.2j, 0.6, 0.3j, -0.8],
            ],
            dtype=complex,
        )
    }

    projected = space.expand(space.compress(density))

    np.testing.assert_allclose(
        projected[(0,)][:2, 2:],
        np.array([[0.0, 0.1 + 0.2j], [-0.1 - 0.2j, 0.0]]),
        atol=1e-14,
    )
    full_bdg_density = assemble_bdg_tb(
        {(0,): projected[(0,)][:2, :2]},
        {(0,): projected[(0,)][:2, 2:]},
        ndof=2,
    )
    validate_bdg_tb(full_bdg_density, ndof=2, ndim=1, name="BdG correction")


def test_scalar_onsite_bdg_pairing_is_rejected_or_projected_to_zero():
    invalid = {
        (0,): np.array(
            [[0.0, 0.4], [0.4, -0.0]],
            dtype=complex,
        )
    }
    with pytest.raises(ValueError, match="Delta"):
        validate_bdg_tb(invalid, ndof=1, ndim=1, name="BdG correction")

    model = Model(
        {(0,): np.zeros((1, 1), dtype=complex)},
        {(0,): np.ones((1, 1), dtype=complex)},
        filling=1.0,
        kT=0.1,
        superconducting=True,
    )
    space = ActiveDensitySpace.bdg(model)
    projected = space.expand(space.compress(invalid))

    assert space.num_params == 1
    assert projected[(0,)][0, 1] == pytest.approx(0.0)
    validate_bdg_tb(projected, ndof=1, ndim=1, name="BdG correction")


def test_bdg_meanfield_density_space_roundtrip():
    interaction = {
        (0,): np.ones((2, 2), dtype=complex),
        (1,): np.array([[0.0, 2.0], [0.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 0.0], [2.0, 0.0]], dtype=complex),
    }
    model = Model(
        spinful_chain(),
        interaction,
        filling=1.0,
        kT=0.1,
        superconducting=True,
    )
    space = ActiveDensitySpace.bdg(model)
    guess = guess_tb([(0,), (1,), (-1,)], 2, superconducting=True)

    projected = space.project(guess)
    density = space.expand(space.compress(projected))

    np.testing.assert_allclose(density[(0,)], projected[(0,)])
    validate_bdg_tb(
        assemble_bdg_tb(
            {key: block[:2, :2] for key, block in density.items()},
            {key: block[:2, 2:] for key, block in density.items()},
            ndof=2,
        ),
        ndof=2,
        ndim=1,
        name="BdG correction",
    )


def test_bdg_correction_assembly_validates_particle_hole_structure():
    normal = {(0,): np.diag([0.1, -0.2]).astype(complex)}
    anomalous = {(0,): np.array([[0.0, 0.3], [-0.3, 0.0]], dtype=complex)}
    model = type("M", (), {"_ndof": 2, "_ndim": 1})()

    correction = assemble_bdg_correction(normal, anomalous, model)

    validate_bdg_tb(correction, ndof=2, ndim=1, name="BdG correction")


def test_bdg_tb_validation_rejects_nonantisymmetric_pairing():
    normal = {(0,): np.zeros((2, 2), dtype=complex)}
    anomalous = {(0,): np.array([[0.0, 0.2], [0.2, 0.0]], dtype=complex)}
    bad = assemble_bdg_tb(normal, anomalous, ndof=2)

    with pytest.raises(ValueError, match="Delta"):
        validate_bdg_tb(bad, ndof=2, ndim=1, name="BdG correction")


def test_spatial_symmetry_can_force_unsupported_active_entries_to_zero():
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    model = Model(
        spinful_chain(),
        {(0,): np.diag([1.0, 0.0]).astype(complex)},
        filling=1.0,
        kT=0.1,
        spatial_symmetries=(
            SpatialSymmetry(
                lattice_matrix=np.eye(1, dtype=int),
                unitaries_by_shift={(0,): swap},
            ),
        ),
    )

    with pytest.warns(UserWarning, match="outside the h_int active support"):
        space = ActiveDensitySpace.normal(model)

    assert space.num_params == 0
    np.testing.assert_allclose(space.expand(np.empty(0))[(0,)], np.zeros((2, 2)))


def test_spatial_symmetry_with_orbital_shifts_reduces_required_entries():
    project_a = np.diag([1.0, 0.0]).astype(complex)
    project_b = np.diag([0.0, 1.0]).astype(complex)
    h_int = {
        (0,): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        (1,): np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex),
    }
    model = Model(
        spinful_chain(),
        h_int,
        filling=1.0,
        kT=0.1,
        spatial_symmetries=(
            SpatialSymmetry(
                lattice_matrix=np.eye(1, dtype=int),
                unitaries_by_shift={(0,): project_a, (1,): project_b},
            ),
        ),
    )

    with pytest.warns(UserWarning, match="outside the h_int active support"):
        space = ActiveDensitySpace.normal(model)
    params = np.arange(space.num_params, dtype=float)
    compressed = space.compress_from_entries(
        space.required_coordinates.values_from_tb(space.expand(params))
    )

    assert len(space.required_realspace_entries()) <= len(space.active_entries)
    np.testing.assert_allclose(compressed, params)


def test_antiunitary_spatial_symmetry_constrains_active_values_to_real():
    h_int = {(0,): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)}
    model = Model(
        spinful_chain(),
        h_int,
        filling=1.0,
        kT=0.1,
        spatial_symmetries=(
            SpatialSymmetry(
                lattice_matrix=np.eye(1, dtype=int),
                unitaries_by_shift={(0,): np.eye(2, dtype=complex)},
                antiunitary=True,
            ),
        ),
    )

    space = ActiveDensitySpace.normal(model)
    active_density = space.expand(np.ones(space.num_params))

    for block in active_density.values():
        np.testing.assert_allclose(block.imag, np.zeros_like(block.imag), atol=1e-12)


@pytest.mark.parametrize(
    ("tb", "nk"),
    [
        (_full_grid_tb(dim=1, max_order=3, matrix_size=2, seed=0), 16),
        (_full_grid_tb(dim=2, max_order=2, matrix_size=2, seed=1), 8),
    ],
    ids=("full_grid_1d", "full_grid_2d"),
)
def test_fourier_roundtrip_on_representative_models(tb, nk):
    ndim = len(next(iter(tb)))
    kham = tb_to_kgrid(tb, nk=nk)
    recovered = ifftn_to_tb(np.fft.ifftn(kham, axes=np.arange(ndim)))

    for key, matrix in tb.items():
        assert np.allclose(recovered[key], matrix)

    extra_keys = set(recovered) - set(tb)
    for key in extra_keys:
        assert np.allclose(recovered[key], np.zeros_like(recovered[key]))


@pytest.mark.parametrize(
    ("builder", "nk"),
    [(spinful_chain, 12), (qiwuzhang, 8)],
    ids=("spinful_chain_1d", "qiwuzhang_2d"),
)
def test_kfunc_matches_sampled_kgrid_on_representative_models(builder, nk):
    tb = builder()
    ndim = len(next(iter(tb)))
    kham = tb_to_kgrid(tb, nk=nk)
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    shifted = np.concatenate((ks[nk // 2 :], ks[: nk // 2]))
    points = np.array(list(it.product(*([shifted] * ndim))))
    sampled = tb_to_kfunc(tb)(points).reshape(kham.shape)

    assert np.allclose(kham, sampled)
