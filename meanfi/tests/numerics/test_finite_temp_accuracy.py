import warnings

import numpy as np
import pytest
import scipy.sparse as sparse

from meanfi import (
    FermiSimplex,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
)
import meanfi.density.kpoint.matrix_functions.rational.scheme as rational_matrix_functions
from meanfi.density.kpoint.matrix_functions.rational import PreparedMumpsRationalNode
from meanfi.density.kpoint.matrix_functions.rational.common import SparseRationalLayout
from meanfi.tests.fixtures.models import (
    max_density_error,
    spinful_chain,
)


pytestmark = [pytest.mark.numerics, pytest.mark.perf_slow]


def _sparse_tb(tb):
    return {key: sparse.csr_matrix(value) for key, value in tb.items()}


def test_zero_dimensional_normal_rational_rejects_dense_matrix():
    tb_zero_dim = {tuple(): np.diag([-1.0, 1.0]).astype(complex)}
    keys = [tuple()]
    matrix_function = RationalFOE(initial_poles=4, max_poles=256)

    with pytest.raises(ValueError, match="RationalFOE is supported only for sparse"):
        density_matrix_at_mu(
            tb_zero_dim,
            mu=0.1,
            kT=0.15,
            keys=keys,
            integration=UniformGrid(
                nk=128,
                matrix_function=matrix_function,
            ),
        )

    with pytest.raises(ValueError, match="RationalFOE is supported only for sparse"):
        density_matrix(
            tb_zero_dim,
            filling=0.9,
            kT=0.15,
            keys=keys,
            integration=UniformGrid(
                nk=128,
                matrix_function=matrix_function,
            ),
            filling_tol=1e-2,
            mu_tol=1e-8,
        )


@pytest.mark.parametrize(
    ("matrix_function", "atol"),
    [
        (None, 1e-2),
        (RationalFOE(initial_poles=4, max_poles=128), 1e-2),
    ],
    ids=["default-sparse-aaa", "explicit-aaa"],
)
@pytest.mark.usefixtures("require_mumps")
def test_sparse_normal_rational_matches_direct_reference_at_mu(matrix_function, atol):
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            matrix_function=DirectDiagonalization(),
        ),
    )
    result = density_matrix_at_mu(
        sparse_tb,
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            matrix_function=matrix_function,
        ),
    )

    actual_density_error = max_density_error(result.to_tb(), reference.to_tb())
    assert abs(result.mu) <= 1e-12
    assert actual_density_error <= atol
    assert result.errors.density_matrix_integration is None


@pytest.mark.usefixtures("require_mumps")
def test_sparse_normal_rational_fixed_filling_matches_dense_reference():
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
    )
    result = density_matrix(
        sparse_tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert abs(result.mu - reference.mu) <= 2e-2
    assert abs(result.filling - reference.filling) <= 1e-2
    assert max_density_error(result.to_tb(), reference.to_tb()) <= 2e-2


@pytest.mark.parametrize(
    ("matrix_function", "atol"),
    [
        (None, 2e-2),
        (RationalFOE(initial_poles=4, max_poles=128), 2e-2),
    ],
    ids=["default-sparse-aaa", "explicit-aaa"],
)
@pytest.mark.usefixtures("require_mumps")
def test_sparse_periodic_grid_matches_dense_reference_at_mu(matrix_function, atol):
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=31,
            matrix_function=DirectDiagonalization(),
        ),
    )
    result = density_matrix_at_mu(
        sparse_tb,
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=31,
            matrix_function=matrix_function,
        ),
    )

    assert abs(result.mu) <= 1e-12
    assert max_density_error(result.to_tb(), reference.to_tb()) <= atol


@pytest.mark.usefixtures("require_mumps")
def test_sparse_periodic_grid_fixed_filling_matches_dense_reference():
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=31,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
    )
    result = density_matrix(
        sparse_tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=31,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_charge_evaluations=80,
    )

    assert abs(result.mu - reference.mu) <= 2e-2
    assert abs(result.filling - reference.filling) <= 1e-2
    assert max_density_error(result.to_tb(), reference.to_tb()) <= 2e-2


@pytest.mark.usefixtures("require_mumps")
def test_normal_scf_sparse_minimal_selection_matches_dense_reference():
    dense_h0 = spinful_chain()
    dense_hint = {(0,): np.diag([1.2, 0.0]).astype(complex)}
    sparse_h0 = _sparse_tb(dense_h0)
    sparse_hint = {(0,): sparse.csr_matrix(dense_hint[(0,)])}

    integration = UniformGrid(
        nk=128,
    )
    model_sparse = Model(sparse_h0, sparse_hint, filling=1.0, kT=0.15)

    space = model_sparse._space

    dense_result = density_matrix(
        dense_h0,
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=integration,
        filling_tol=1e-2,
    )
    sparse_result = density_matrix(
        sparse_h0,
        filling=1.0,
        kT=0.15,
        coordinates=space.required_coordinates,
        integration=integration,
        filling_tol=1e-2,
    )
    assert abs(dense_result.mu - sparse_result.mu) <= 5e-4
    assert abs(dense_result.filling - sparse_result.filling) <= 5e-4
    np.testing.assert_allclose(
        sparse_result.values,
        dense_result.values_for(space.required_coordinates),
        atol=5e-4,
    )


def test_dtype_controls_are_validated():
    with pytest.raises(ValueError, match="dtype must be complex64 or complex128"):
        UniformGrid(nk=128, dtype="float32")

    assert "dtype" not in FermiSimplex.__dataclass_fields__


@pytest.mark.usefixtures("require_mumps")
def test_periodic_complex64_matches_complex128():
    tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    high_precision = density_matrix(
        tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            dtype="complex128",
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )
    low_precision = density_matrix(
        tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=128,
            dtype="complex64",
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert abs(low_precision.mu - high_precision.mu) <= 5e-3
    assert max_density_error(low_precision.to_tb(), high_precision.to_tb()) <= 2e-2


@pytest.mark.usefixtures("require_mumps")
def test_sparse_solver_result_does_not_expose_reduced_density():
    h0 = {(0,): sparse.csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=complex))}
    h_int = {(0,): sparse.csr_matrix(np.diag([1.0, 1.0]).astype(complex))}
    model = Model(h0, h_int, filling=1.0, kT=0.15)
    result = solver(
        model,
        {(0,): sparse.csr_matrix(np.zeros((2, 2), dtype=complex))},
        integration=UniformGrid(
            nk=128,
        ),
        scf=LinearMixing(max_iterations=1, alpha=0.5),
        scf_tol=1.0,
        filling_tol=1e-2,
    )

    assert not hasattr(result, "density_matrix")
    assert not hasattr(result, "density_matrix_result")


def test_density_postprocessing_returns_complete_dense_blocks():
    h0 = {(0,): np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=complex)}
    h_int = {(0,): np.diag([1.0, 1.0]).astype(complex)}
    model = Model(h0, h_int, filling=1.0, kT=0.15)
    result = solver(
        model,
        {(0,): np.zeros((2, 2), dtype=complex)},
        integration=UniformGrid(
            nk=128,
        ),
        scf=LinearMixing(max_iterations=1, alpha=0.5),
        scf_tol=1.0,
        filling_tol=1e-2,
    )
    density = density_matrix_at_mu(
        model.hamiltonian_from_meanfield(result.mean_field),
        result.mu,
        kT=model.kT,
        keys=[(0,)],
        integration=UniformGrid(
            nk=128,
        ),
    )
    onsite_block = density.to_tb()[(0,)]
    assert not np.allclose(np.diag(np.diag(onsite_block)), onsite_block, atol=1e-12)


def test_sparse_aaa_terms_certify_scalar_error_on_local_interval():
    terms = rational_matrix_functions._aaa_terms_for_interval(
        512,
        lower=-2.0,
        upper=2.5,
        kT=0.2,
        initial_poles=4,
        scalar_tolerance=1e-2,
    )
    probe = np.linspace(-2.0, 2.5, 2001, dtype=float)
    approximation = rational_matrix_functions._evaluate_canonical_rational(
        probe,
        constant=terms.constant,
        shifts=terms.shifts,
        residues=terms.residues,
    )
    target = 1.0 / (1.0 + np.exp(probe / 0.2))
    assert np.max(np.abs(target - approximation)) <= 1e-2


@pytest.mark.parametrize("kT", [0.002, 0.2, 2.0])
def test_sparse_aaa_certifies_hot_and_cold_spectra(kT):
    from scipy.special import expit

    terms = rational_matrix_functions._aaa_terms_for_interval(
        128,
        lower=-3.0,
        upper=3.0,
        kT=kT,
        scalar_tolerance=1e-9,
    )
    probe = np.unique(
        np.r_[np.linspace(-3.0, 3.0, 20001), np.linspace(-10 * kT, 10 * kT, 10001)]
    )
    probe = probe[(probe >= -3.0) & (probe <= 3.0)]
    approximation = rational_matrix_functions._evaluate_canonical_rational(
        probe,
        constant=terms.constant,
        shifts=terms.shifts,
        residues=terms.residues,
    )
    np.testing.assert_allclose(approximation, expit(-probe / kT), atol=1e-9, rtol=0)


def test_sparse_aaa_interval_cache_reuses_nested_interval_fit():
    shared_cache = []
    matrix = sparse.csr_matrix(np.array([[0.2, -1.0], [-1.0, -0.1]], dtype=complex))
    model = Model(
        {(0,): np.asarray(matrix.toarray(), dtype=complex)},
        {(0,): np.ones((2, 2), dtype=complex)},
        filling=1.0,
        kT=0.15,
    )
    space = model._space
    node = PreparedMumpsRationalNode(
        matrix,
        kT=0.15,
        q_diag=np.ones(2, dtype=float),
        options=RationalFOE(initial_poles=4, max_poles=128),
        charge_tolerance=1e-2,
        layout=SparseRationalLayout.build(
            density_coordinates=space.required_coordinates,
            trace_weights_diag=np.ones(2, dtype=float),
            include_all_diagonal=False,
        ),
        matrix_function_tol=1e-2,
        shared_aaa_interval_cache=shared_cache,
    )

    first = node._sparse_terms(0.0)
    cache_size = len(shared_cache)
    second = node._sparse_terms(0.0)

    assert first is second
    assert len(shared_cache) == cache_size


@pytest.mark.usefixtures("require_mumps")
def test_strained_graphene_single_shot_sparse_aaa_is_stable():
    pytest.importorskip("kwant")
    from docs.source.tutorial.scripts.zero_temp_validation import (
        _build_strained_graphene_inputs,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        h0, _h_int, guess, filling, _data, _k_path = _build_strained_graphene_inputs()
    result = density_matrix(
        {key: value for key, value in h0.items()},
        filling=filling,
        kT=0.2,
        keys=[(0, 0)],
        integration=UniformGrid(
            nk=4,  # Large sparse smoke test; accuracy is checked on small models.
            matrix_function=RationalFOE(initial_poles=4, max_poles=128),
        ),
        filling_tol=1e-1,
        mu_tol=1e-8,
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - filling) <= 1e-1
