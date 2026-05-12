import numpy as np
import pytest

from meanfi import (
    AdaptiveSimplex,
    AndersonMixing,
    Model,
    UniformGrid,
    add_tb,
    density_matrix,
    expectation_value,
    solver,
)
from meanfi.density.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.interop import kwant as utils


pytestmark = [pytest.mark.numerics, pytest.mark.perf_slow]
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def _build_graphene_bad_point():
    kwant = pytest.importorskip("kwant")

    s0 = np.identity(2)
    sx = np.array([[0, 1], [1, 0]])
    sz = np.diag([1, -1])

    graphene = kwant.lattice.general(
        [(1, 0), (1 / 2, np.sqrt(3) / 2)],
        [(0, 0), (0, 1 / np.sqrt(3))],
        norbs=2,
    )
    a, b = graphene.sublattices
    bulk = kwant.Builder(kwant.TranslationalSymmetry(*graphene.prim_vecs))
    bulk[a.shape((lambda pos: True), (0, 0))] = 0 * s0
    bulk[b.shape((lambda pos: True), (0, 0))] = 0 * s0
    bulk[graphene.neighbors(1)] = s0

    h0 = utils.builder_to_tb(bulk)

    def onsite_int(site, U):
        del site
        return U * sx

    def nn_int(site1, site2, V):
        del site1, site2
        return V * np.ones((2, 2))

    builder_int = utils.build_interacting_syst(
        builder=bulk,
        lattice=graphene,
        func_onsite=onsite_int,
        func_hop=nn_int,
        max_neighbor=1,
    )
    h_int = utils.builder_to_tb(
        builder_int,
        {"U": 3.111111111111111, "V": 0.3333333333333333},
    )
    return h0, h_int, sz


def _sdw_measure(h0, mf, sz):
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    s_list = [sx, sy, np.diag([1, -1])]
    rho = density_matrix(
        add_tb(h0, mf),
        filling=2,
        integration=UniformGrid(nk=40),
        keys=[(0, 0)],
        filling_tol=1e-6,
    ).density_matrix
    sdw_sq = 0.0
    for spin_matrix in s_list:
        operator = {(0, 0): np.kron(sz, spin_matrix)}
        sdw_sq += float(np.abs(expectation_value(rho, operator)) ** 2)
    return sdw_sq


def _broad_hermitian_correction(keys, ndof: int, *, seed: int):
    rng = np.random.RandomState(seed)
    correction = {}
    for key in keys:
        key = tuple(key)
        if key in correction:
            continue
        matrix = rng.rand(ndof, ndof) * np.exp(2j * np.pi * rng.rand(ndof, ndof))
        opposite = tuple(-component for component in key)
        if key == opposite:
            correction[key] = 0.5 * (matrix + matrix.conj().T)
        else:
            correction[key] = matrix
            correction[opposite] = matrix.conj().T
    return correction


@requires_ext
def test_refinement_depth_improves_bad_graphene_point_diagnostic():
    h0, h_int, sz = _build_graphene_bad_point()
    ndof = len(next(iter(h0.values())))
    seeds = [0, 2, 3]

    def solve_sdw_measure(refinement_depth: int):
        values = []
        for seed in seeds:
            model = Model(h0, h_int, filling=2)
            with pytest.warns(UserWarning, match="projected away"):
                result = solver(
                    model,
                    _broad_hermitian_correction(h_int, ndof, seed=seed),
                    integration=AdaptiveSimplex(
                        density_matrix_tol=1e-4,
                        refinement_depth=refinement_depth,
                    ),
                    scf=AndersonMixing(M=0, line_search="wolfe", max_iterations=1000),
                    scf_tol=1e-7,
                    filling_tol=1e-3,
                )
            values.append(_sdw_measure(h0, result.mf, sz))
        return values

    depth0 = solve_sdw_measure(0)
    depth1 = solve_sdw_measure(1)

    # Diagnostic only: the refined path should avoid collapsing to the
    # symmetry-preserving branch at this known ill-behaved point.
    assert min(depth0) > 0.5
    assert min(depth1) > 0.5
