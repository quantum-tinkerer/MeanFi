import itertools as it

import numpy as np
import pytest
from scipy import sparse

from meanfi.model import Model
from meanfi.space import (
    DensityCoordinates,
    SpatialSymmetry,
)
from meanfi.space.coordinates import canonical_tb_keys
from meanfi.space.reducers import real_to_complex
from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import compare_dicts
from meanfi.tb.transforms import ifftn_to_tb, tb_to_kfunc, tb_to_kgrid
from meanfi.tests.fixtures.models import qiwuzhang, spinful_chain


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
    )
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


def test_active_density_space_required_entries_roundtrip():
    model = Model(
        spinful_chain(),
        {(0,): np.diag([1.0, 0.0]).astype(complex)},
        filling=1.0,
        kT=0.1,
    )
    space = model.scf_space
    values = np.array([0.25 + 0.0j])

    params = space.params_from_required_entries(values)
    recovered = space.required_coordinates.values_from_tb(
        space.meanfield_input_from_params(params)
    )

    assert space.required_coordinates.entries == (((0,), 0, 0),)
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
    space = model.scf_space
    density = {
        (0,): np.array([[0.3, 0.2j], [-0.2j, 1.7]], dtype=complex),
        (1,): np.array([[0.4, 0.5], [0.6, 0.7]], dtype=complex),
        (-1,): np.array([[0.4, 0.6], [0.5, 0.7]], dtype=complex),
    }

    projected = space.project_meanfield_input(density)
    recovered = space.meanfield_input_from_params(
        space.params_from_meanfield_input(projected)
    )
    compare_dicts(projected, recovered)


def test_no_symmetry_normal_space_uses_compact_orbits_for_full_onsite_support(
    monkeypatch,
):
    from meanfi.space.reducers import OrbitReducer

    def reject_dense_basis(*args, **kwargs):
        raise AssertionError("compact parametrization must not construct a dense basis")

    monkeypatch.setattr(OrbitReducer, "basis", reject_dense_basis)
    ndof = 32
    h_0 = {(0,): np.zeros((ndof, ndof), dtype=complex)}
    h_int = {(0,): np.ones((ndof, ndof), dtype=complex)}

    model = Model(h_0, h_int, filling=1.0, kT=0.1)
    space = model.scf_space

    assert space.num_params == ndof * ndof
    assert len(space.active_coordinates.entries) == ndof * ndof
    assert len(space.required_coordinates.entries) == ndof * (ndof + 1) // 2

    params = np.arange(space.num_params, dtype=float)
    density = space.meanfield_input_from_params(params)
    np.testing.assert_allclose(density[(0,)], density[(0,)].conj().T)
    np.testing.assert_allclose(space.params_from_meanfield_input(density), params)


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("spatial_symmetry", [False, True])
@pytest.mark.parametrize(
    "sparse_h0,sparse_interaction",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_active_space_reconstruction_preserves_sparse_model_inputs(
    superconducting, spatial_symmetry, sparse_h0, sparse_interaction
):
    h0 = spinful_chain()
    interaction = {(0,): np.ones((2, 2), dtype=complex)}
    symmetries = (
        (
            SpatialSymmetry(
                lattice_matrix=np.eye(1, dtype=int),
                unitaries_by_shift={(0,): np.eye(2, dtype=complex)},
                antiunitary=True,
            ),
        )
        if spatial_symmetry
        else ()
    )
    options = dict(
        filling=1.0,
        kT=0.1,
        superconducting=superconducting,
        spatial_symmetries=symmetries,
    )
    dense_space = Model(h0, interaction, **options).scf_space
    space = Model(
        {key: sparse.csr_matrix(value) for key, value in h0.items()}
        if sparse_h0
        else h0,
        {key: sparse.csr_matrix(value) for key, value in interaction.items()}
        if sparse_interaction
        else interaction,
        **options,
    ).scf_space
    params = np.random.default_rng(7).normal(size=dense_space.num_params)
    expected = dense_space.meanfield_input_from_params(params)
    reconstructed = space.meanfield_input_from_params(params)

    for key, block in reconstructed.items():
        assert sparse.issparse(block) == (sparse_h0 or sparse_interaction)
        actual = block.toarray() if sparse.issparse(block) else block
        np.testing.assert_allclose(actual, expected[key], atol=1e-14)
    np.testing.assert_allclose(space.params_from_meanfield_input(reconstructed), params)


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("spatial_symmetry", [False, True])
def test_empty_active_space_reconstructs_sparse_zero_blocks(
    superconducting, spatial_symmetry
):
    symmetries = (
        (
            SpatialSymmetry(
                lattice_matrix=np.eye(1, dtype=int),
                unitaries_by_shift={(0,): np.eye(2, dtype=complex)},
            ),
        )
        if spatial_symmetry
        else ()
    )
    space = Model(
        spinful_chain(),
        {(0,): sparse.csr_matrix((2, 2))},
        filling=1.0,
        kT=0.1,
        superconducting=superconducting,
        spatial_symmetries=symmetries,
    ).scf_space
    reconstructed = space.meanfield_input_from_params(np.empty(0))

    assert space.num_params == space.required_coordinates.value_count == 0
    assert reconstructed[(0,)].shape == ((4, 4) if superconducting else (2, 2))
    assert sparse.isspmatrix_csr(reconstructed[(0,)])
    assert reconstructed[(0,)].nnz == 0
    assert space.params_from_meanfield_input(reconstructed).size == 0


def test_large_sparse_active_space_never_materializes_dense_blocks(monkeypatch):
    def reject_dense(*args, **kwargs):
        raise AssertionError("sparse SCF space must not materialize dense matrices")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense)
    monkeypatch.setattr(sparse.csr_matrix, "todense", reject_dense)
    ndof = 20_000
    identity = sparse.eye(ndof, format="csr", dtype=complex)
    space = Model({(): identity}, {(): identity}, filling=1.0, kT=0.1).scf_space
    params = np.linspace(0.0, 1.0, space.num_params)
    reconstructed = space.meanfield_input_from_params(params)[()]

    assert space.num_params == ndof
    assert sparse.isspmatrix_csr(reconstructed)
    assert reconstructed.nnz <= ndof
    np.testing.assert_array_equal(reconstructed.diagonal(), params)


def test_large_spatial_symmetry_space_fails_before_allocating_dense_basis(monkeypatch):
    from meanfi.space.reducers import OrbitReducer

    def reject_dense_basis(*args, **kwargs):
        raise AssertionError(
            "dense basis safety limit must be checked before allocation"
        )

    monkeypatch.setattr(OrbitReducer, "basis", reject_dense_basis)
    ndof = 128
    with pytest.raises(MemoryError, match="above.*safety limit"):
        Model(
            {(): np.eye(ndof)},
            {(): np.ones((ndof, ndof))},
            filling=1.0,
            kT=0.1,
            spatial_symmetries=(
                SpatialSymmetry(
                    lattice_matrix=np.empty((0, 0), dtype=int),
                    unitaries_by_shift={(): np.eye(ndof)},
                ),
            ),
        )


def test_bdg_space_imposes_strict_particle_hole_reduction():
    model = Model(
        spinful_chain(),
        {(0,): np.ones((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.1,
        superconducting=True,
    )
    space = model.scf_space
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

    projected = space.meanfield_input_from_params(
        space.params_from_meanfield_input(density)
    )

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
    space = model.scf_space
    projected = space.meanfield_input_from_params(
        space.params_from_meanfield_input(invalid)
    )

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
    space = model.scf_space
    guess = model.random_meanfield(rng=0)

    projected = space.project_meanfield_input(guess)
    density = space.meanfield_input_from_params(
        space.params_from_meanfield_input(projected)
    )

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
    correction = assemble_bdg_tb(normal, anomalous, ndof=2)

    validate_bdg_tb(correction, ndof=2, ndim=1, name="BdG correction")


def test_bdg_tb_validation_rejects_nonantisymmetric_pairing():
    normal = {(0,): np.zeros((2, 2), dtype=complex)}
    anomalous = {(0,): np.array([[0.0, 0.2], [0.2, 0.0]], dtype=complex)}
    bad = assemble_bdg_tb(normal, anomalous, ndof=2)

    with pytest.raises(ValueError, match="Delta"):
        validate_bdg_tb(bad, ndof=2, ndim=1, name="BdG correction")


def test_spatial_symmetry_can_force_unsupported_active_entries_to_zero():
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    with pytest.warns(UserWarning, match="outside the h_int active support"):
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
        space = model.scf_space

    assert space.num_params == 0
    np.testing.assert_allclose(
        space.meanfield_input_from_params(np.empty(0))[(0,)],
        np.zeros((2, 2)),
    )


def test_spatial_symmetry_with_orbital_shifts_reduces_required_entries():
    project_a = np.diag([1.0, 0.0]).astype(complex)
    project_b = np.diag([0.0, 1.0]).astype(complex)
    h_int = {
        (0,): np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        (1,): np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex),
    }
    with pytest.warns(UserWarning, match="outside the h_int active support"):
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
        space = model.scf_space
    params = np.arange(space.num_params, dtype=float)
    compressed = space.params_from_required_entries(
        space.required_coordinates.values_from_tb(
            space.meanfield_input_from_params(params)
        )
    )

    assert len(space.required_coordinates.entries) <= len(
        space.active_coordinates.entries
    )
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

    space = model.scf_space
    active_density = space.meanfield_input_from_params(np.ones(space.num_params))

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
    kham = tb_to_kgrid(tb, (nk,) * ndim)
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
    kham = tb_to_kgrid(tb, (nk,) * ndim)
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    shifted = np.concatenate((ks[nk // 2 :], ks[: nk // 2]))
    points = np.array(list(it.product(*([shifted] * ndim))))
    sampled = tb_to_kfunc(tb)(points).reshape(kham.shape)

    assert np.allclose(kham, sampled)


@pytest.mark.parametrize("shape", [(), (1,), (5,), (8,), (4, 6), (3, 4, 5)])
def test_fourier_preserves_all_modes_and_hermiticity(shape):
    from meanfi import kgrid_to_tb
    from meanfi.tb.validate import validate_hermiticity

    rng = np.random.default_rng(21)
    raw = rng.normal(size=(*shape, 2, 2)) + 1j * rng.normal(size=(*shape, 2, 2))
    grid = raw + raw.conj().swapaxes(-1, -2)
    tb = kgrid_to_tb(grid)
    validate_hermiticity(tb)
    np.testing.assert_allclose(tb_to_kgrid(tb, shape), grid, atol=1e-14)
    points = np.array(list(it.product(*(2 * np.pi * np.fft.fftfreq(n) for n in shape))))
    sampled = tb_to_kfunc(tb)(points).reshape(grid.shape)
    np.testing.assert_allclose(sampled, grid, atol=1e-14)


@pytest.mark.parametrize("mixed", [False, True])
def test_sparse_fourier_matches_dense_without_densifying_kfunc_inputs(
    monkeypatch, mixed
):
    tb = spinful_chain()
    sparse_tb = {key: sparse.csr_matrix(matrix) for key, matrix in tb.items()}
    if mixed:
        sparse_tb[(0,)] = tb[(0,)]
    np.testing.assert_allclose(tb_to_kgrid(sparse_tb, (7,)), tb_to_kgrid(tb, (7,)))

    def reject_dense(*args, **kwargs):
        raise AssertionError("kfunc must accumulate sparse entries directly")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense)
    k = np.array([[-0.7], [0.0], [0.3]])
    np.testing.assert_allclose(tb_to_kfunc(sparse_tb)(k), tb_to_kfunc(tb)(k))
