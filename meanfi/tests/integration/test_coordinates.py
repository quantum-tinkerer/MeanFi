import builtins

import numpy as np
import pytest
from scipy import sparse

import meanfi.space.coordinates as coordinates_module
from meanfi.space.coordinates import DensityCoordinates, full_density_coordinates


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "sparse_type",
    [sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix, sparse.csr_array],
)
def test_sparse_gather_matches_dense_in_coordinate_order(sparse_type, monkeypatch):
    coordinates = DensityCoordinates.from_pairs(
        size=3,
        keys=[(1,), (0,), (-1,)],
        pairs_by_key={
            (1,): (np.array([2, 0, 1]), np.array([0, 2, 1])),
            (0,): (np.array([1]), np.array([2])),
            (-1,): (np.array([0]), np.array([1])),
        },
    )
    matrix = np.array([[1, 0, 2j], [0, 0, 3], [4 - 1j, 0, 5]])
    expected = coordinates.values_from_tb({(1,): matrix, (0,): matrix})
    blocks = {(1,): sparse_type(matrix), (0,): sparse_type(matrix)}

    def unexpected_dense(*args, **kwargs):
        pytest.fail("selected sparse extraction must not form a dense matrix")

    monkeypatch.setattr(sparse_type, "toarray", unexpected_dense)
    actual = coordinates.values_from_tb(blocks)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [4 - 1j, 2j, 0, 3, 0])


def test_sparse_reconstruction_matches_dense_and_preserves_selected_zeros():
    coordinates = DensityCoordinates.from_pairs(
        size=3,
        keys=[(1,), (0,), (-1,)],
        pairs_by_key={
            (1,): (np.array([2, 0]), np.array([0, 2])),
            (0,): (np.array([1]), np.array([1])),
        },
    )
    values = np.array([2j, 0, 0.5])
    dense = coordinates.values_to_tb(values)
    compact = coordinates.values_to_tb(values, sparse=True)
    for key in coordinates.keys:
        assert sparse.isspmatrix_csr(compact[key])
        np.testing.assert_array_equal(compact[key].toarray(), dense[key])
    assert compact[(1,)].nnz == 2
    assert compact[(-1,)].nnz == 0
    np.testing.assert_array_equal(coordinates.values_from_tb(compact), values)


def test_large_selected_layout_never_allocates_or_enumerates_the_full_matrix(
    monkeypatch,
):
    size = 100_000
    coordinates = DensityCoordinates.from_pairs(
        size=size,
        keys=[(0,)],
        pairs_by_key={(0,): (np.array([size - 1, 0]), np.array([0, size - 1]))},
    )
    values = np.array([0.25j, 0.75])
    original_zeros = np.zeros

    def guarded_range(*args):
        result = builtins.range(*args)
        if len(result) >= size:
            pytest.fail("selected layouts must not enumerate all matrix entries")
        return result

    def unexpected_dense(*args, **kwargs):
        pytest.fail("selected sparse operations must not form a dense matrix")

    def guarded_zeros(shape, *args, **kwargs):
        if shape == (size, size):
            pytest.fail("selected sparse operations must not allocate a dense block")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(coordinates_module, "range", guarded_range, raising=False)
    monkeypatch.setattr(np, "zeros", guarded_zeros)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", unexpected_dense)
    assert coordinates.is_full is False
    blocks = coordinates.values_to_tb(values, sparse=True)
    block = blocks[(0,)]
    assert block.nnz == 2
    assert block.data.nbytes + block.indices.nbytes + block.indptr.nbytes < 1_000_000
    np.testing.assert_array_equal(coordinates.values_from_tb(blocks), values)


@pytest.mark.parametrize("keys", [[], [(0,)]])
def test_empty_layouts_are_valid_and_roundtrip(keys):
    from_pairs = DensityCoordinates.from_pairs(size=3, keys=keys, pairs_by_key={})
    from_entries = DensityCoordinates.from_entries(size=3, keys=keys, entries=())
    for coordinates in (from_pairs, from_entries):
        assert coordinates.value_count == 0
        assert coordinates.entries == ()
        for use_sparse in (False, True):
            blocks = coordinates.values_to_tb(np.empty(0), sparse=use_sparse)
            assert list(blocks) == keys
            np.testing.assert_array_equal(coordinates.values_from_tb(blocks), [])


def test_empty_full_layout_does_not_build_matrix_coordinates(monkeypatch):
    def unexpected_grid(*args, **kwargs):
        pytest.fail("no keys require no matrix coordinates")

    monkeypatch.setattr(np, "meshgrid", unexpected_grid)
    coordinates = full_density_coordinates([], size=100_000)
    assert coordinates.value_count == 0
    assert coordinates.is_full


def test_from_entries_keeps_key_order_and_sorts_unique_pairs():
    coordinates = DensityCoordinates.from_entries(
        size=2,
        keys=[(1,), (0,)],
        entries=(((0,), 1, 0), ((1,), 1, 1), ((1,), 0, 1), ((1,), 1, 1)),
    )
    assert coordinates.entries == (((1,), 0, 1), ((1,), 1, 1), ((0,), 1, 0))


def test_duplicate_layout_pairs_or_keys_are_rejected():
    with pytest.raises(ValueError, match="unique within each key"):
        DensityCoordinates.from_pairs(
            size=2,
            keys=[(0,)],
            pairs_by_key={(0,): (np.array([1, 1]), np.array([0, 0]))},
        )
    with pytest.raises(ValueError, match="keys must be unique"):
        DensityCoordinates.from_pairs(size=2, keys=[(0,), (0,)], pairs_by_key={})
    with pytest.raises(ValueError, match="entry key is absent"):
        DensityCoordinates.from_entries(size=2, keys=[(0,)], entries=(((1,), 0, 0),))


def test_full_layout_can_use_arbitrary_unique_coordinate_order():
    coordinates = DensityCoordinates.from_pairs(
        size=2,
        keys=[(0,)],
        pairs_by_key={(0,): (np.array([1, 0, 1, 0]), np.array([1, 0, 0, 1]))},
    )
    assert coordinates.is_full
    values = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(
        coordinates.values_from_tb(coordinates.values_to_tb(values, sparse=True)),
        values,
    )
