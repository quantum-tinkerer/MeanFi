import itertools as it

import numpy as np
import pytest

from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.tb.tb import compare_dicts
from meanfi.tb.transforms import ifftn_to_tb, tb_to_kfunc, tb_to_kgrid
from meanfi.tb.utils import generate_tb_keys, guess_tb
from meanfi.tests.helpers import qiwuzhang, spinful_chain


pytestmark = pytest.mark.integration


def _full_grid_tb(*, dim: int, max_order: int, matrix_size: int, seed: int):
    rng = np.random.default_rng(seed)
    keys = [range(-max_order + 1, max_order) for _ in range(dim)]
    return {
        key: rng.normal(size=(matrix_size, matrix_size))
        + 1j * rng.normal(size=(matrix_size, matrix_size))
        for key in it.product(*keys)
    }


@pytest.mark.parametrize(
    ("cutoff", "dim", "ndof", "seed"),
    [(1, 1, 2, 0), (1, 2, 3, 1)],
)
def test_parametrization_roundtrip_on_generated_guesses(cutoff, dim, ndof, seed):
    np.random.seed(seed)
    tb = guess_tb(generate_tb_keys(cutoff, dim), ndof)
    params = tb_to_rparams(tb)
    compare_dicts(tb, rparams_to_tb(params, list(tb), ndof))


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
