import itertools as it

import kwant
import numpy as np
import pytest

from meanfi.interop.kwant import builder_to_tb, tb_to_builder
from meanfi.tb.ops import compare_dicts
from meanfi.tb.utils import generate_tb_keys


def _hermitian_tb(keys, ndof, seed=0):
    rng = np.random.default_rng(seed)
    tb = {}
    for key in keys:
        key = tuple(key)
        if key in tb:
            continue
        matrix = rng.normal(size=(ndof, ndof)) + 1j * rng.normal(size=(ndof, ndof))
        opposite = tuple(-component for component in key)
        if key == opposite:
            tb[key] = 0.5 * (matrix + matrix.conj().T)
        else:
            tb[key] = matrix
            tb[opposite] = matrix.conj().T
    return tb


pytestmark = pytest.mark.integration


def test_kwant_conversion_roundtrip_on_representative_builder():
    lattice = kwant.lattice.general(
        [(1.0, 0.0), (0.4, 1.1)],
        basis=[(0.0, 0.0), (0.2, 0.3)],
        norbs=[1, 2],
    )
    dummy_builder = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    for sublattice in lattice.sublattices:
        dummy_builder[lattice.shape(lambda pos: True, (0.0, 0.0))] = np.eye(
            sublattice.norbs
        )

    tb = _hermitian_tb(generate_tb_keys(1, 2), 3)
    builder = tb_to_builder(
        tb, list(dummy_builder.sites()), dummy_builder.symmetry.periods
    )
    compare_dicts(tb, builder_to_tb(builder))


def test_kwant_builder_order_is_invariant():
    norbs = 2
    nlen = 3
    lattice = kwant.lattice.square(norbs=norbs)
    builder1 = kwant.Builder(kwant.TranslationalSymmetry((nlen, 0), (0, nlen)))
    builder2 = kwant.Builder(kwant.TranslationalSymmetry((nlen, 0), (0, nlen)))

    sites = [lattice(i, j) for i in range(nlen) for j in range(nlen)]
    values = [np.diag([idx + 1, idx + 2]).astype(float) for idx in range(len(sites))]
    for site, value in zip(sites, values, strict=True):
        builder1[site] = value

    indices = np.arange(len(sites))
    np.random.default_rng(0).shuffle(indices)
    for index in indices:
        builder2[sites[index]] = values[index]

    compare_dicts(builder_to_tb(builder1), builder_to_tb(builder2))


def test_kwant_supercell_callable_roundtrip():
    lattice = kwant.lattice.general(
        [(1.0, 0.0), (0.25, 1.0)],
        basis=[(0.0, 0.0), (0.5, 0.25)],
        norbs=[1, 2],
    )

    def random_matrix_digest(n: int, m: int, salt: int):
        matrix = np.zeros((n, m))
        for row, col in it.product(range(n), range(m)):
            matrix[row, col] = kwant.digest.uniform(f"{n}-{m}-{row}-{col}-{salt}")
        return matrix

    def onsite(site, alpha, beta):
        n = site.family.norbs
        amplitude = alpha * random_matrix_digest(n, n, 0)
        phase = 1j * 2.0 * np.pi * beta * random_matrix_digest(n, n, 1)
        value = amplitude * phase
        return value + value.conj().T

    def hopping(site1, site2, gamma, delta):
        n1 = site1.family.norbs
        n2 = site2.family.norbs
        amplitude = gamma * random_matrix_digest(n1, n2, 2)
        phase = 1j * 2.0 * np.pi * delta * random_matrix_digest(n1, n2, 3)
        return amplitude * phase

    builder = kwant.Builder(kwant.TranslationalSymmetry(*(2 * lattice.prim_vecs)))
    for sublattice in lattice.sublattices:
        builder[lattice.shape(lambda pos: True, (0.0, 0.0))] = onsite
    builder[lattice.neighbors()] = hopping

    params = {"alpha": 0.2, "beta": 0.3, "gamma": 0.4, "delta": 0.1}
    tb, data = builder_to_tb(builder, params=params, return_data=True)
    rebuilt = tb_to_builder(tb, data["sites"], data["periods"])

    for site1, site2 in it.product(data["sites"], repeat=2):
        if site1 == site2:
            expected = builder[site1](
                site=site1, alpha=params["alpha"], beta=params["beta"]
            )
            assert np.allclose(expected, rebuilt[site1])
            continue

        try:
            expected = builder[site1, site2](
                site1,
                site2,
                gamma=params["gamma"],
                delta=params["delta"],
            )
        except KeyError:
            continue
        assert np.allclose(expected, rebuilt[site1, site2])
