import inspect

import numpy as np
import pytest

import meanfi.integrate.matrix_functions.prepared_normal as prepared_normal
import meanfi.integrate.quadrature.normal_backend as normal_quadrature_backend
from meanfi import (
    AdaptiveQuadrature,
    ChebyshevFOE,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
    density_matrix,
    density_matrix_at_mu,
    fermi_dirac,
    solver,
)
from meanfi.tests.helpers import (
    assert_estimator_covers_actual,
    converged_dense_reference,
    local_two_band_2d,
    max_density_error,
    max_density_estimate,
    spinful_chain,
)


pytestmark = pytest.mark.numerics


def _supports_prepared_payloads() -> bool:
    try:
        from stateful_quadrature import StatefulIntegrator
    except ImportError:
        return False
    return "payload_builder" in inspect.signature(StatefulIntegrator).parameters


requires_prepared_payloads = pytest.mark.skipif(
    not _supports_prepared_payloads(),
    reason="prepared-payload stateful_quadrature support is unavailable",
)


def _sparse_tb(tb):
    sparse = pytest.importorskip("scipy.sparse")
    return {key: sparse.csr_matrix(value) for key, value in tb.items()}


def _normal_matrix_function_cases():
    return [
        pytest.param(
            ChebyshevFOE(initial_order=8, max_order=256),
            id="chebyshev",
        ),
        pytest.param(
            RationalFOE(initial_poles=4, max_poles=64),
            id="rational",
        ),
    ]


def test_finite_temperature_local_two_band_matches_exact_reference_across_tolerance_ladder(
    density_tolerance_ladder,
    scalar_tolerance_ladder,
):
    tb = local_two_band_2d()
    keys = [(0, 0), (1, 0), (0, 1)]
    exact_rho = {
        (0, 0): np.diag(fermi_dirac(np.array([-1.0, 1.0]), 0.2, 0.0)),
        (1, 0): np.zeros((2, 2)),
        (0, 1): np.zeros((2, 2)),
    }

    for density_atol, scalar_tol in zip(
        density_tolerance_ladder,
        scalar_tolerance_ladder,
        strict=True,
    ):
        result = density_matrix(
            tb,
            filling=1.0,
            kT=0.2,
            keys=keys,
            integration=AdaptiveQuadrature(density_matrix_tol=density_atol),
            filling_tol=scalar_tol,
        )
        actual_density_error = max_density_error(result.density_matrix, exact_rho)

        assert abs(result.mu) <= scalar_tol
        assert abs(result.filling - 1.0) <= scalar_tol
        assert actual_density_error <= density_atol
        assert_estimator_covers_actual(
            actual_density_error,
            max_density_estimate(result.density_matrix_error),
        )
        assert result.filling_residual is not None
        assert result.filling_residual <= scalar_tol


def test_finite_temperature_density_matrix_at_mu_matches_self_converged_reference_across_density_ladder(
    density_tolerance_ladder,
):
    tb = spinful_chain()
    keys = [(0,), (1,), (-1,)]
    reference = converged_dense_reference(
        tb,
        mu=0.0,
        kT=0.15,
        keys=keys,
        target_tol=min(density_tolerance_ladder) / 10.0,
        nk_start=2001,
        nk_max=16001,
    )

    for density_atol in density_tolerance_ladder:
        result = density_matrix_at_mu(
            tb,
            mu=0.0,
            kT=0.15,
            keys=keys,
            integration=AdaptiveQuadrature(density_matrix_tol=density_atol),
        )
        actual_density_error = max_density_error(result.density_matrix, reference.rho)

        assert actual_density_error <= density_atol
        assert result.info.error_estimate_available is True
        assert_estimator_covers_actual(
            actual_density_error,
            max_density_estimate(result.density_matrix_error),
        )


def test_finite_temperature_fixed_filling_matches_self_converged_reference_across_tolerance_ladder(
    density_tolerance_ladder,
    scalar_tolerance_ladder,
):
    tb = spinful_chain()
    keys = [(0,), (1,), (-1,)]
    filling = 0.7
    reference = converged_dense_reference(
        tb,
        filling=filling,
        kT=0.15,
        keys=keys,
        target_tol=min(
            min(density_tolerance_ladder),
            min(scalar_tolerance_ladder),
        )
        / 10.0,
        nk_start=2001,
        nk_max=16001,
    )

    for density_atol, scalar_tol in zip(
        density_tolerance_ladder,
        scalar_tolerance_ladder,
        strict=True,
    ):
        result = density_matrix(
            tb,
            filling=filling,
            kT=0.15,
            keys=keys,
            integration=AdaptiveQuadrature(density_matrix_tol=density_atol),
            filling_tol=scalar_tol,
            mu_tol=scalar_tol,
        )
        actual_density_error = max_density_error(result.density_matrix, reference.rho)
        actual_charge_error = abs(result.filling - filling)

        assert actual_density_error <= density_atol
        assert actual_charge_error <= scalar_tol
        assert result.info.error_estimate_available is True


@requires_prepared_payloads
@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_normal_matrix_functions_match_direct_reference_at_mu(
    matrix_function,
    sparse_input,
):
    dense_tb = spinful_chain()
    tb = _sparse_tb(dense_tb) if sparse_input else dense_tb
    keys = [(0,), (1,), (-1,)]

    reference = density_matrix_at_mu(
        dense_tb,
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            max_refinements=80,
            matrix_function=DirectDiagonalization(),
        ),
    )
    result = density_matrix_at_mu(
        tb,
        mu=0.0,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=8e-4,
            max_refinements=80,
            matrix_function=matrix_function,
        ),
    )

    actual_density_error = max_density_error(result.density_matrix, reference.density_matrix)
    assert actual_density_error <= 4e-3
    assert np.isfinite(result.filling)


@requires_prepared_payloads
@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_normal_matrix_functions_match_direct_reference_at_fixed_filling(
    matrix_function,
    sparse_input,
):
    dense_tb = spinful_chain()
    tb = _sparse_tb(dense_tb) if sparse_input else dense_tb
    keys = [(0,), (1,), (-1,)]

    reference = density_matrix(
        dense_tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            max_refinements=80,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
    )
    result = density_matrix(
        tb,
        filling=0.7,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=80,
            matrix_function=matrix_function,
        ),
        filling_tol=1e-4,
        mu_tol=1e-8,
    )

    actual_density_error = max_density_error(result.density_matrix, reference.density_matrix)
    assert actual_density_error <= 5e-3
    assert abs(result.filling - 0.7) <= 1e-4
    assert abs(result.mu - reference.mu) <= 5e-3


@requires_prepared_payloads
@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
def test_normal_matrix_function_path_does_not_fallback_to_exact_diagonalization(
    monkeypatch,
    matrix_function,
):
    def fail_if_exact(*args, **kwargs):
        raise AssertionError("Prepared normal path should not call exact diagonalization")

    monkeypatch.setattr(normal_quadrature_backend, "spectral_payload", fail_if_exact)
    result = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=matrix_function,
        ),
    )

    assert np.isfinite(result.filling)


@requires_prepared_payloads
@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
def test_sparse_normal_matrix_function_path_avoids_dense_sparse_conversion(
    monkeypatch,
    matrix_function,
):
    sparse = pytest.importorskip("scipy.sparse")
    original_asarray = prepared_normal.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if sparse.issparse(value):
            raise AssertionError("Sparse normal matrix-function path should not densify matrices")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(prepared_normal.np, "asarray", guarded_asarray)
    result = density_matrix_at_mu(
        _sparse_tb(spinful_chain()),
        mu=0.0,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=matrix_function,
        ),
    )

    assert np.isfinite(result.filling)


@requires_prepared_payloads
@pytest.mark.parametrize(
    ("attr_name", "matrix_function"),
    [
        pytest.param(
            "PreparedNormalChebyshevNode",
            ChebyshevFOE(initial_order=8, max_order=256),
            id="chebyshev",
        ),
        pytest.param(
            "PreparedNormalRationalNode",
            RationalFOE(initial_poles=4, max_poles=64),
            id="rational",
        ),
    ],
)
def test_normal_matrix_function_payloads_are_reused_within_fixed_filling_solve(
    monkeypatch,
    attr_name,
    matrix_function,
):
    created = 0
    base_class = getattr(prepared_normal, attr_name)

    class CountingPreparedNode(base_class):
        def __init__(self, *args, **kwargs):
            nonlocal created
            created += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(normal_quadrature_backend, attr_name, CountingPreparedNode)
    result = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=60,
            matrix_function=matrix_function,
        ),
        filling_tol=1e-4,
        mu_tol=1e-8,
    )

    assert result.info.charge_integration_calls is not None
    assert result.info.charge_integration_calls >= 2
    assert created == result.info.unique_evals


@requires_prepared_payloads
@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
def test_sparse_normal_matrix_function_scf_smoke(matrix_function):
    sparse = pytest.importorskip("scipy.sparse")
    h_0_sparse = _sparse_tb(spinful_chain())
    h_int_sparse = {(0,): sparse.csr_matrix(np.zeros((2, 2), dtype=complex))}
    guess = {(0,): sparse.csr_matrix(np.zeros((2, 2), dtype=complex))}
    model = Model(h_0_sparse, h_int_sparse, filling=1.0, kT=0.15)

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=matrix_function,
        ),
        scf=LinearMixing(max_iterations=2, alpha=0.5),
        scf_tol=1e-8,
        filling_tol=1e-4,
    )

    assert abs(result.density_matrix_result.filling - 1.0) <= 1e-4
    assert np.isfinite(result.density_matrix_result.mu)
    assert result.info.iterations <= 2
