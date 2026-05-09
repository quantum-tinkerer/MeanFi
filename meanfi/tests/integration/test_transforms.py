import itertools as it

import numpy as np
import pytest

from meanfi.space.density_selection import density_selection_from_pairs
from meanfi.space.hermitian import normal_density_selection
from meanfi.space.particlehole import (
    bdg_density_selection_from_top_half,
    bdg_top_half_selection,
)
from meanfi.space.bdg import (
    bdg_density_to_rparams,
    bdg_tb_to_rparams,
    rparams_to_bdg_density,
    rparams_to_bdg_tb,
)
from meanfi.space.params import canonical_tb_keys, real_to_complex
from meanfi.space.normal import (
    rparams_to_tb,
    tb_to_rparams,
)
from meanfi.space import MeanFieldDensitySpace
from meanfi.meanfield import assemble_bdg_correction
from meanfi.model import Model
from meanfi.tb.bdg import validate_bdg_tb
from meanfi.tb.ops import compare_dicts
from meanfi.tb.transforms import ifftn_to_tb, tb_to_kfunc, tb_to_kgrid
from meanfi.tb.utils import generate_tb_keys, guess_tb
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


def _bdg_examples():
    yield (
        {(0,): np.array([[0.0, 0.3], [0.3, 0.0]], dtype=complex)},
        1,
    )
    yield (
        {
            (1,): np.array([[0.0, 0.2], [-0.2, 0.0]], dtype=complex),
            (-1,): np.array([[0.0, -0.2], [0.2, 0.0]], dtype=complex),
            (0,): np.array([[0.4, 0.0], [0.0, -0.4]], dtype=complex),
        },
        1,
    )
    anomalous = np.array([[0.25, 0.08], [0.08, 0.18]], dtype=complex)
    yield (
        {
            (0,): np.block(
                [[np.diag([0.1, -0.2]), anomalous], [anomalous, -np.diag([0.1, -0.2])]]
            )
        },
        2,
    )
    yield (
        {
            (1, 0): np.array([[0.0, 0.22], [0.22, 0.0]], dtype=complex),
            (-1, 0): np.array([[0.0, 0.22], [0.22, 0.0]], dtype=complex),
            (0, 1): np.array([[0.0, -0.22], [-0.22, 0.0]], dtype=complex),
            (0, -1): np.array([[0.0, -0.22], [-0.22, 0.0]], dtype=complex),
            (0, 0): np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex),
        },
        1,
    )


@pytest.mark.parametrize(
    ("cutoff", "dim", "ndof", "seed"),
    [(1, 1, 2, 0), (1, 2, 3, 1)],
)
def test_parametrization_roundtrip_on_generated_guesses(cutoff, dim, ndof, seed):
    np.random.seed(seed)
    tb = guess_tb(generate_tb_keys(cutoff, dim), ndof)
    params = tb_to_rparams(tb)
    compare_dicts(tb, rparams_to_tb(params, list(tb), ndof))


def test_canonical_tb_keys_are_deterministic_and_explicit():
    keys = [(1, 0), (0, 0), (0, -1), (0, 1), (-1, 0)]

    assert canonical_tb_keys(keys) == [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def test_density_selection_value_order_and_negative_grid_key():
    selection = density_selection_from_pairs(
        size=2,
        keys=[(0,), (-1,)],
        selected_pairs={
            (0,): (np.array([0]), np.array([0])),
            (-1,): (np.array([1]), np.array([0])),
        },
        allow_empty=False,
    )
    assert selection is not None
    assert selection.value_count == 2
    np.testing.assert_array_equal(selection.all_rows, np.array([0, 1]))
    np.testing.assert_array_equal(selection.all_cols, np.array([0, 0]))
    assert [key_selection.key for key_selection in selection.key_selections] == [
        (0,),
        (-1,),
    ]
    assert selection.key_slice((-1,)) == slice(1, 2)

    values = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    selected_tb = selection.values_to_tb(values)
    assert selected_tb[(0,)][0, 0] == values[0]
    assert selected_tb[(-1,)][1, 0] == values[1]
    np.testing.assert_allclose(selection.values_from_tb({(0,): selected_tb[(0,)]}), [values[0], 0.0])
    np.testing.assert_allclose(selection.values_from_tb(selected_tb), values)

    real_space_values = np.zeros((4, selection.value_count), dtype=complex)
    real_space_values[-1, selection.key_slice((-1,))] = values[1]
    kgrid_values = np.fft.fftn(real_space_values, axes=(0,))
    selected_from_grid = _selected_value_grid_to_tb(selection, kgrid_values, ndim=1)
    assert selected_from_grid[(-1,)][1, 0] == pytest.approx(values[1])


def test_real_to_complex_rejects_odd_length_values():
    with pytest.raises(ValueError, match="even number"):
        real_to_complex(np.array([1.0, 2.0, 3.0]))


def test_parametrization_rejects_asymmetric_selection():
    with pytest.raises(ValueError, match="symmetric under key inversion"):
        rparams_to_tb(np.zeros(6), [(0,), (1,)], 1)

    with pytest.raises(ValueError, match="symmetric under key inversion"):
        canonical_tb_keys([(0,), (1,)])


def test_parametrization_roundtrip_on_multi_orbital_selection():
    tb = {
        (0,): np.array([[0.2, 0.1 + 0.3j], [0.1 - 0.3j, -0.4]], dtype=complex),
        (1,): np.array([[0.5 + 0.2j, -0.2], [0.3j, 0.1 - 0.1j]], dtype=complex),
        (-1,): np.array([[0.5 - 0.2j, -0.3j], [-0.2, 0.1 + 0.1j]], dtype=complex),
    }

    params = tb_to_rparams(tb)
    compare_dicts(tb, rparams_to_tb(params, list(tb), ndof=2))


def test_selection_aware_normal_parametrization_roundtrip():
    selection = normal_density_selection(
        keys=[(0,), (1,), (-1,)],
        interaction_tb={
            (0,): np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
            (1,): np.array([[0.0, 0.0], [2.0, 0.0]], dtype=complex),
            (-1,): np.array([[0.0, 2.0], [0.0, 0.0]], dtype=complex),
        },
        ndof=2,
        local_key=(0,),
        allow_empty=True,
    )
    assert selection is not None

    tb = {
        (0,): np.array([[0.3, 0.2j], [-0.2j, 1.7]], dtype=complex),
        (1,): np.array([[0.4, 0.5], [0.6, 0.7]], dtype=complex),
        (-1,): np.array([[0.4, 0.6], [0.5, 0.7]], dtype=complex),
    }

    params = tb_to_rparams(tb, selection=selection)
    assert params.size < tb_to_rparams(tb).size
    recovered = rparams_to_tb(params, list(tb), ndof=2, selection=selection)

    np.testing.assert_allclose(
        recovered[(0,)], np.array([[0.3, 0.0], [0.0, 1.7]], dtype=complex)
    )
    np.testing.assert_allclose(
        recovered[(1,)], np.array([[0.0, 0.0], [0.6, 0.0]], dtype=complex)
    )
    np.testing.assert_allclose(recovered[(-1,)], recovered[(1,)].conj().T)


def test_normal_meanfield_density_space_roundtrip_and_meanfield():
    interaction = {
        (0,): np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        (1,): np.array([[0.0, 0.0], [2.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 2.0], [0.0, 0.0]], dtype=complex),
    }
    model = Model(spinful_chain(), interaction, filling=1.0, kT=0.1)
    space = MeanFieldDensitySpace.normal(model)
    density = {
        (0,): np.array([[0.3, 0.2j], [-0.2j, 1.7]], dtype=complex),
        (1,): np.array([[0.4, 0.5], [0.6, 0.7]], dtype=complex),
        (-1,): np.array([[0.4, 0.6], [0.5, 0.7]], dtype=complex),
    }

    projected = space.project_guess(density)
    recovered = space.density_from_params(space.params_from_density(projected))
    compare_dicts(projected, recovered)

    correction = space.meanfield_from_density(projected, mu=0.25)
    assert model._local_key in correction
    assert correction[model._local_key].shape == (model._ndof, model._ndof)


@pytest.mark.parametrize(("bdg_tb", "ndof"), list(_bdg_examples()))
def test_bdg_parametrization_roundtrip_on_representative_states(bdg_tb, ndof):
    params = bdg_tb_to_rparams(bdg_tb, ndof)
    recovered = rparams_to_bdg_tb(params, list(bdg_tb), ndof)

    validate_bdg_tb(
        recovered, ndof=ndof, ndim=len(next(iter(bdg_tb))), name="BdG correction"
    )
    compare_dicts(bdg_tb, recovered)


def test_selection_aware_bdg_parametrization_and_density_roundtrip():
    interaction = {
        (0,): np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        (1,): np.array([[0.0, 2.0], [0.0, 0.0]], dtype=complex),
        (-1,): np.array([[0.0, 0.0], [2.0, 0.0]], dtype=complex),
    }
    selection = bdg_top_half_selection(
        keys=[(0,), (1,), (-1,)],
        interaction_tb=interaction,
        ndof=2,
        local_key=(0,),
    )
    density_selection = bdg_density_selection_from_top_half(selection, ndof=2)
    assert density_selection.keys == selection.electron.keys
    assert density_selection.value_count == (
        selection.electron.value_count + selection.anomalous.value_count
    )

    normal_block = {
        (0,): np.array([[0.2, 0.3j], [-0.3j, 0.8]], dtype=complex),
        (1,): np.array([[0.4, 0.5], [0.6, 0.7]], dtype=complex),
        (-1,): np.array([[0.4, 0.6], [0.5, 0.7]], dtype=complex),
    }
    anomalous_block = {
        (0,): np.array([[0.0, 0.1], [0.2, 0.0]], dtype=complex),
        (1,): np.array([[0.8, 0.9], [1.0, 1.1]], dtype=complex),
        (-1,): np.array([[1.1, 1.0], [0.9, 0.8]], dtype=complex),
    }
    bdg_tb = assemble_bdg_correction(
        normal_block, anomalous_block, type("M", (), {"_ndof": 2})()
    )

    params = bdg_tb_to_rparams(bdg_tb, 2, selection=selection)
    assert params.size < bdg_tb_to_rparams(bdg_tb, 2).size
    recovered = rparams_to_bdg_tb(params, list(bdg_tb), 2, selection=selection)
    validate_bdg_tb(recovered, ndof=2, ndim=1, name="BdG correction")

    assert recovered[(0,)][0, 0] == pytest.approx(0.2)
    assert recovered[(0,)][0, 1] == pytest.approx(0.0)
    assert recovered[(1,)][0, 1] == pytest.approx(0.5)
    assert recovered[(1,)][0, 2] == pytest.approx(0.0)
    assert recovered[(1,)][0, 3] == pytest.approx(0.9)

    density = {key: np.array(value, copy=True) for key, value in recovered.items()}
    density_params = bdg_density_to_rparams(density, selection=selection, ndof=2)
    density_recovered = rparams_to_bdg_density(density_params, selection=selection, ndof=2)
    np.testing.assert_allclose(density_recovered[(0,)][:2, :2], recovered[(0,)][:2, :2])
    np.testing.assert_allclose(density_recovered[(1,)][:2, 2:], recovered[(1,)][:2, 2:])


def test_bdg_meanfield_density_space_roundtrip_and_meanfield():
    interaction = {
        (0,): np.eye(2, dtype=complex),
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
    space = MeanFieldDensitySpace.bdg(model)
    guess = guess_tb([(0,), (1,), (-1,)], 2, superconducting=True)

    projected = space.project_guess(guess)
    density = space.density_from_params(space.params_from_density(projected))
    correction = space.meanfield_from_density(density)

    validate_bdg_tb(correction, ndof=2, ndim=1, name="BdG correction")
    assert set(correction) == set(space.active_keys)


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
