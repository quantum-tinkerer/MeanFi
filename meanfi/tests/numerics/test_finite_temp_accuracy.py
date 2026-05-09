import warnings

import numpy as np
import pytest
import scipy.sparse as sparse

from meanfi import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
)
from meanfi.space.hermitian import normal_density_selection
import meanfi.density.kpoint.matrix_functions.rational as rational_matrix_functions
import meanfi.density.integrate.normal as normal_integration
import meanfi.density.integrate.quadrature.runtime as quadrature_runtime
from meanfi.space.normal import tb_to_rparams
from meanfi.scf.normal import (
    _density_update_for_normal_hamiltonian as _evaluate_density_for_hamiltonian,
)
from meanfi.tests.fixtures.models import (
    assert_estimator_covers_actual,
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
            integration=AdaptiveQuadrature(
                density_matrix_tol=1e-2,
                matrix_function=matrix_function,
            ),
        )

    with pytest.raises(ValueError, match="RationalFOE is supported only for sparse"):
        density_matrix(
            tb_zero_dim,
            filling=0.9,
            kT=0.15,
            keys=keys,
            integration=AdaptiveQuadrature(
                density_matrix_tol=1e-2,
                matrix_function=matrix_function,
            ),
            filling_tol=1e-2,
            mu_tol=1e-8,
        )


@pytest.mark.parametrize(
    ("matrix_function", "atol"),
    [
        (None, 1e-2),
        (RationalFOE(initial_poles=4, max_poles=128, rational_scheme="aaa"), 1e-2),
        (RationalFOE(initial_poles=4, max_poles=128, rational_scheme="ozaki"), 2e-2),
    ],
    ids=["default-sparse-aaa", "explicit-aaa", "explicit-ozaki"],
)
def test_sparse_normal_rational_matches_direct_reference_at_mu(matrix_function, atol):
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            matrix_function=DirectDiagonalization(),
        ),
    )
    result = density_matrix_at_mu(
        sparse_tb,
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=40,
            matrix_function=matrix_function,
        ),
    )

    actual_density_error = max_density_error(
        result.density_matrix, reference.density_matrix
    )
    assert abs(result.mu) <= 1e-12
    assert actual_density_error <= atol
    assert_estimator_covers_actual(
        actual_density_error,
        max(
            float(np.max(np.abs(block)))
            for block in result.density_matrix_error.values()
        ),
    )


def test_sparse_normal_rational_fixed_filling_matches_dense_reference():
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
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
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=40,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert abs(result.mu - reference.mu) <= 2e-2
    assert abs(result.filling - reference.filling) <= 1e-2
    assert max_density_error(result.density_matrix, reference.density_matrix) <= 2e-2


@pytest.mark.parametrize(
    ("matrix_function", "atol"),
    [
        (None, 2e-2),
        (RationalFOE(initial_poles=4, max_poles=128, rational_scheme="aaa"), 2e-2),
        (RationalFOE(initial_poles=4, max_poles=128, rational_scheme="ozaki"), 3e-2),
    ],
    ids=["default-sparse-aaa", "explicit-aaa", "explicit-ozaki"],
)
def test_sparse_uniform_grid_matches_dense_reference_at_mu(matrix_function, atol):
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=31,
            density_matrix_tol=1e-8,
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
            density_matrix_tol=1e-2,
            matrix_function=matrix_function,
        ),
    )

    assert abs(result.mu) <= 1e-12
    assert max_density_error(result.density_matrix, reference.density_matrix) <= atol


def test_sparse_uniform_grid_fixed_filling_matches_dense_reference():
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=UniformGrid(
            nk=31,
            density_matrix_tol=1e-8,
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
            density_matrix_tol=1e-2,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_charge_evaluations=80,
    )

    assert abs(result.mu - reference.mu) <= 2e-2
    assert abs(result.filling - reference.filling) <= 1e-2
    assert max_density_error(result.density_matrix, reference.density_matrix) <= 2e-2


def test_normal_scf_sparse_minimal_selection_matches_dense_reference():
    dense_h0 = spinful_chain()
    dense_hint = {(0,): np.diag([1.2, 0.0]).astype(complex)}
    sparse_h0 = _sparse_tb(dense_h0)
    sparse_hint = {(0,): sparse.csr_matrix(dense_hint[(0,)])}

    integration = AdaptiveQuadrature(
        density_matrix_tol=1e-2,
        max_refinements=40,
    )
    model_dense = Model(dense_h0, dense_hint, filling=1.0, kT=0.15)
    model_sparse = Model(sparse_h0, sparse_hint, filling=1.0, kT=0.15)

    selection = normal_density_selection(
        keys=[(0,)],
        interaction_tb=dense_hint,
        ndof=2,
        local_key=(0,),
        allow_empty=True,
    )
    assert selection is not None

    dense_result = _evaluate_density_for_hamiltonian(
        model_dense,
        dense_h0,
        keys=[(0,)],
        integration=integration,
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_charge_evaluations=None,
        mu_guess=0.0,
        density_selection=None,
    )
    sparse_result = _evaluate_density_for_hamiltonian(
        model_sparse,
        sparse_h0,
        keys=[(0,)],
        integration=integration,
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_charge_evaluations=None,
        mu_guess=0.0,
        density_selection=selection,
    )

    assert abs(dense_result.mu - sparse_result.mu) <= 5e-4
    assert abs(dense_result.filling - sparse_result.filling) <= 5e-4
    np.testing.assert_allclose(
        tb_to_rparams(dense_result.density_matrix, selection=selection),
        tb_to_rparams(sparse_result.density_matrix, selection=selection),
        atol=5e-4,
    )


def test_workspace_precision_controls_are_validated():
    with pytest.raises(ValueError, match="workspace_precision must be 64 or 128"):
        AdaptiveQuadrature(workspace_precision=32)

    with pytest.raises(
        ValueError,
        match="AdaptiveSimplex currently supports only workspace_precision=128",
    ):
        density_matrix(
            spinful_chain(),
            filling=1.0,
            kT=0.0,
            keys=[(0,), (1,), (-1,)],
            integration=AdaptiveSimplex(workspace_precision=64),
        )


def test_quadrature_workspace_precision_64_matches_128():
    tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    high_precision = density_matrix(
        tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=60,
            workspace_precision=128,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )
    low_precision = density_matrix(
        tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=60,
            workspace_precision=64,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert abs(low_precision.mu - high_precision.mu) <= 5e-3
    assert (
        max_density_error(low_precision.density_matrix, high_precision.density_matrix)
        <= 2e-2
    )


def test_solver_density_result_zeroes_entries_outside_interaction_tb():
    h0 = {(0,): sparse.csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=complex))}
    h_int = {(0,): sparse.csr_matrix(np.diag([1.0, 1.0]).astype(complex))}
    model = Model(h0, h_int, filling=1.0, kT=0.15)
    result = solver(
        model,
        {(0,): sparse.csr_matrix(np.zeros((2, 2), dtype=complex))},
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=40,
        ),
        scf=LinearMixing(max_iterations=1, alpha=0.5),
        scf_tol=1e-8,
        filling_tol=1e-2,
    )

    onsite_block = result.density_matrix_result.density_matrix[(0,)]
    np.testing.assert_allclose(np.diag(np.diag(onsite_block)), onsite_block, atol=1e-12)


def test_dense_solver_density_result_keeps_full_dense_blocks():
    h0 = {(0,): np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=complex)}
    h_int = {(0,): np.diag([1.0, 1.0]).astype(complex)}
    model = Model(h0, h_int, filling=1.0, kT=0.15)
    result = solver(
        model,
        {(0,): np.zeros((2, 2), dtype=complex)},
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=40,
        ),
        scf=LinearMixing(max_iterations=1, alpha=0.5),
        scf_tol=1e-8,
        filling_tol=1e-2,
    )
    onsite_block = result.density_matrix_result.density_matrix[(0,)]
    assert not np.allclose(np.diag(np.diag(onsite_block)), onsite_block, atol=1e-12)


def test_fixed_filling_rational_density_pass_uses_frozen_charge_mesh(monkeypatch):
    calls = []
    original_run_integrator = quadrature_runtime.run_integrator

    def wrapped_run_integrator(*args, **kwargs):
        result = original_run_integrator(*args, **kwargs)
        calls.append(
            {
                "status": result.status,
                "max_subdivisions": kwargs["max_subdivisions"],
                "n_kernel_evals": int(result.n_kernel_evals),
                "subdivisions": int(getattr(result, "subdivisions", 0)),
            }
        )
        return result

    monkeypatch.setattr(quadrature_runtime, "run_integrator", wrapped_run_integrator)
    monkeypatch.setattr(normal_integration, "run_integrator", wrapped_run_integrator)
    result = density_matrix(
        _sparse_tb(spinful_chain()),
        filling=0.7,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=60,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert len(calls) >= 2
    assert calls[-1]["max_subdivisions"] == 0
    assert calls[-1]["status"] in {"converged", "max_subdivisions"}
    assert calls[-1]["n_kernel_evals"] == 0
    assert calls[-1]["subdivisions"] == 0
    assert result.info.n_kernel_evals == result.info.unique_evals


def test_sparse_aaa_terms_certify_scalar_error_on_local_interval():
    terms, _builder = rational_matrix_functions._aaa_terms_for_interval(
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
        tail_lower_bound=terms.tail_lower_bound,
        tail_upper_bound=terms.tail_upper_bound,
    )
    target = 1.0 / (1.0 + np.exp(probe / 0.2))
    assert np.max(np.abs(target - approximation)) <= 1e-2


def test_sparse_aaa_arrowhead_poles_match_barycentric_form():
    builder = rational_matrix_functions._fit_barycentric_weights(
        np.linspace(-2.0, 2.0, 33, dtype=float),
        np.asarray(
            1.0 / (1.0 + np.exp(np.linspace(-2.0, 2.0, 33, dtype=float) / 0.2)),
            dtype=complex,
        ),
        [0, 8, 16, 24, 32],
    )
    probe = np.linspace(-2.0, 2.0, 513, dtype=float)
    barycentric = rational_matrix_functions._barycentric_evaluate(
        probe,
        builder.support_x,
        builder.support_y,
        builder.weights,
    )
    terms, _scalar_error, _gap = rational_matrix_functions._aaa_terms_from_builder(
        builder,
        lower=-2.0,
        upper=2.0,
        scalar_tolerance=1e-2,
        tail_lower_bound=None,
        tail_upper_bound=None,
    )
    pole_form = rational_matrix_functions._evaluate_canonical_rational(
        probe,
        constant=terms.constant,
        shifts=terms.shifts,
        residues=terms.residues,
        tail_lower_bound=terms.tail_lower_bound,
        tail_upper_bound=terms.tail_upper_bound,
    )
    np.testing.assert_allclose(pole_form, barycentric, atol=1e-10, rtol=1e-10)


def test_sparse_aaa_interval_cache_reuses_nested_interval_fit():
    shared_cache = []
    matrix = sparse.csr_matrix(np.array([[0.2, -1.0], [-1.0, -0.1]], dtype=complex))
    selection = normal_density_selection(
        keys=[(0,)],
        interaction_tb={(0,): np.ones((2, 2), dtype=complex)},
        ndof=2,
        local_key=(0,),
        allow_empty=True,
    )
    assert selection is not None
    node = rational_matrix_functions.PreparedMumpsRationalNode(
        matrix,
        kT=0.15,
        q_diag=np.ones(2, dtype=float),
        options=RationalFOE(initial_poles=4, max_poles=128, rational_scheme="aaa"),
        charge_tolerance=1e-2,
        density_selection=selection,
        density_tolerance=1e-2,
        trace_weights_diag=np.ones(2, dtype=float),
        shared_aaa_interval_cache=shared_cache,
    )

    first = node._sparse_terms(0.0, pole_count=128, scalar_tolerance=1e-3)
    cache_size = len(shared_cache)
    second = node._sparse_terms(0.0, pole_count=128, scalar_tolerance=1e-3)

    assert first.support_count == second.support_count
    assert len(shared_cache) == cache_size


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
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-1,
            max_refinements=20,
            matrix_function=RationalFOE(
                initial_poles=4, max_poles=128, rational_scheme="aaa"
            ),
        ),
        filling_tol=1e-1,
        mu_tol=1e-8,
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - filling) <= 1e-1
    assert result.info.n_leaves == 1
