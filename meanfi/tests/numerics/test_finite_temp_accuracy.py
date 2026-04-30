import inspect

import numpy as np
import pytest

from meanfi.core.matrix import matrix_bound, to_dense
from meanfi.integrate.density_support import (
    bdg_density_entry_support,
    normal_density_entry_support,
)
import meanfi.integrate.matrix_functions.common as matrix_function_common
import meanfi.integrate.matrix_functions.prepared_normal as prepared_normal
import meanfi.integrate.matrix_functions.rational as rational_matrix_functions
import meanfi.integrate.quadrature.normal_backend as normal_quadrature_backend
import meanfi.integrate.quadrature.runtime as quadrature_runtime
from meanfi.integrate.quadrature.payloads import build_tb_payload_helpers, tb_k_matrix
from meanfi.params.rparams import tb_to_rparams
from meanfi import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
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


pytestmark = [pytest.mark.numerics, pytest.mark.perf_slow]


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
        pytest.param(ChebyshevFOE(initial_order=8, max_order=256), id="chebyshev"),
        pytest.param(RationalFOE(initial_poles=4, max_poles=256), id="rational"),
    ]


def _loop_chebyshev_coefficients(
    order: int,
    *,
    center: float,
    scale: float,
    kT: float,
    mu: float,
    oversampling: int,
    derivative: bool,
) -> np.ndarray:
    n_nodes = max(64, int(oversampling) * (order + 1))
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    energies = scale * np.cos(theta) + center
    occupation = fermi_dirac(energies, kT, mu)
    if derivative:
        values = occupation * (1.0 - occupation) / kT
    else:
        values = occupation

    coeffs = np.empty(order + 1, dtype=float)
    coeffs[0] = np.mean(values)
    for mode in range(1, order + 1):
        coeffs[mode] = 2.0 * np.mean(values * np.cos(mode * theta))
    return coeffs


def _sparse_square_supercell(
    *,
    lx: int = 4,
    ly: int = 4,
    flux: float = 0.09,
    onsite_x_amp: float = 0.28,
    onsite_y_amp: float = 0.18,
    onsite_mix_amp: float = 0.10,
):
    sparse = pytest.importorskip("scipy.sparse")

    n_sites = lx * ly
    onsite = sparse.dok_matrix((n_sites, n_sites), dtype=complex)
    x_hop = sparse.dok_matrix((n_sites, n_sites), dtype=complex)
    y_hop = sparse.dok_matrix((n_sites, n_sites), dtype=complex)

    def site(x: int, y: int) -> int:
        return y * lx + x

    for y in range(ly):
        for x in range(lx):
            index = site(x, y)
            onsite[index, index] = (
                onsite_x_amp * np.cos(2.0 * np.pi * (x + 0.3) / lx)
                + onsite_y_amp * np.sin(2.0 * np.pi * (y + 0.2) / ly)
                + onsite_mix_amp * np.cos(2.0 * np.pi * (x + 2.0 * y + 0.5) / max(lx, ly))
            )
            phase_y = np.exp(1j * flux * x)
            neighbor_x = site((x + 1) % lx, y)
            neighbor_y = site(x, (y + 1) % ly)
            if x + 1 < lx:
                onsite[index, neighbor_x] = -1.0
            else:
                x_hop[index, neighbor_x] = -1.0
            if y + 1 < ly:
                onsite[index, neighbor_y] = -phase_y
            else:
                y_hop[index, neighbor_y] = -phase_y

    return {
        (0, 0): (onsite + onsite.conjugate().transpose()).tocsr(),
        (1, 0): x_hop.tocsr(),
        (-1, 0): x_hop.conjugate().transpose().tocsr(),
        (0, 1): y_hop.tocsr(),
        (0, -1): y_hop.conjugate().transpose().tocsr(),
    }


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


@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
@pytest.mark.parametrize("density_atol", [1e-1, 1e-2], ids=["tol-1e-1", "tol-1e-2"])
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_normal_matrix_functions_match_direct_reference_at_mu(
    matrix_function,
    density_atol,
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
            density_matrix_tol=density_atol,
            max_refinements=80,
            matrix_function=matrix_function,
        ),
    )

    actual_density_error = max_density_error(result.density_matrix, reference.density_matrix)
    assert actual_density_error <= density_atol
    assert np.isfinite(result.filling)


@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
@pytest.mark.parametrize("density_atol", [1e-1, 1e-2], ids=["tol-1e-1", "tol-1e-2"])
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "sparse"])
def test_normal_matrix_functions_match_direct_reference_at_fixed_filling(
    matrix_function,
    density_atol,
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
            density_matrix_tol=density_atol,
            max_refinements=80,
            matrix_function=matrix_function,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    actual_density_error = max_density_error(result.density_matrix, reference.density_matrix)
    assert actual_density_error <= max(3.0 * density_atol, 2e-2)
    assert abs(result.filling - 0.7) <= 1e-2
    assert abs(result.mu - reference.mu) <= 5e-2


@pytest.mark.parametrize("matrix_function", _normal_matrix_function_cases())
def test_normal_matrix_function_path_does_not_fallback_to_exact_diagonalization(
    monkeypatch,
    matrix_function,
):
    def fail_if_exact(*args, **kwargs):
        raise AssertionError("Normal matrix-function path should not call exact diagonalization")

    monkeypatch.setattr(normal_quadrature_backend, "spectral_payload", fail_if_exact)
    result = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=40,
            matrix_function=matrix_function,
        ),
    )

    assert np.isfinite(result.filling)


@requires_prepared_payloads
def test_sparse_normal_matrix_function_path_avoids_dense_sparse_conversion(
    monkeypatch,
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
            density_matrix_tol=1e-2,
            max_refinements=40,
            matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
        ),
    )

    assert np.isfinite(result.filling)


def test_sparse_payload_reconstruction_survives_matrix_bound_reordering():
    tb = _sparse_square_supercell()
    kernel, matrix_from_payload = build_tb_payload_helpers(tb)
    point = np.array([0.0, 0.0], dtype=float)
    payload_row = np.array(kernel(point[np.newaxis, :])[0], copy=True)

    matrix_bound(tb[(0, 0)])

    reconstructed = matrix_from_payload(payload_row)
    direct = tb_k_matrix(tb, point)
    diff = reconstructed - direct
    if hasattr(diff, "toarray"):
        diff = diff.toarray()

    assert float(np.max(np.abs(diff))) <= 1e-12


@requires_prepared_payloads
def test_normal_matrix_function_payloads_are_reused_within_fixed_filling_solve(
    monkeypatch,
):
    created = 0
    base_class = prepared_normal.PreparedNormalChebyshevNode

    class CountingPreparedNode(base_class):
        def __init__(self, *args, **kwargs):
            nonlocal created
            created += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        normal_quadrature_backend,
        "PreparedNormalChebyshevNode",
        CountingPreparedNode,
    )
    result = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=60,
            matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert result.info.charge_integration_calls is not None
    assert result.info.charge_integration_calls >= 2
    assert created == result.info.unique_evals
    assert result.info.n_kernel_evals == result.info.unique_evals


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
            density_matrix_tol=1e-2,
            max_refinements=40,
            matrix_function=matrix_function,
        ),
        scf=LinearMixing(max_iterations=2, alpha=0.5),
        scf_tol=1e-8,
        filling_tol=1e-2,
    )

    assert abs(result.density_matrix_result.filling - 1.0) <= 1e-2
    assert np.isfinite(result.density_matrix_result.mu)
    assert result.info.iterations <= 2


@requires_prepared_payloads
def test_normal_scf_sparse_minimal_support_matches_dense_reference():
    sparse = pytest.importorskip("scipy.sparse")
    dense_h0 = spinful_chain()
    dense_hint = {(0,): np.diag([1.2, 0.0]).astype(complex)}
    dense_guess = {(0,): np.diag([0.05, 0.0]).astype(complex)}

    sparse_h0 = _sparse_tb(dense_h0)
    sparse_hint = {(0,): sparse.csr_matrix(dense_hint[(0,)])}
    sparse_guess = {(0,): sparse.csr_matrix(dense_guess[(0,)])}

    integration = AdaptiveQuadrature(
        density_matrix_tol=1e-2,
        max_refinements=40,
        matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
    )
    scf = LinearMixing(max_iterations=16, alpha=0.5)

    dense_result = solver(
        Model(dense_h0, dense_hint, filling=1.0, kT=0.15),
        dense_guess,
        integration=integration,
        scf=scf,
        scf_tol=1e-6,
        filling_tol=1e-2,
    )
    sparse_result = solver(
        Model(sparse_h0, sparse_hint, filling=1.0, kT=0.15),
        sparse_guess,
        integration=integration,
        scf=scf,
        scf_tol=1e-6,
        filling_tol=1e-2,
    )

    assert abs(dense_result.density_matrix_result.mu - sparse_result.density_matrix_result.mu) <= 5e-4
    assert abs(dense_result.density_matrix_result.filling - sparse_result.density_matrix_result.filling) <= 5e-4
    assert abs(dense_result.info.residual_norm - sparse_result.info.residual_norm) <= 5e-4
    for key in dense_result.mf:
        np.testing.assert_allclose(
            to_dense(dense_result.mf[key]),
            to_dense(sparse_result.mf[key]),
            atol=5e-4,
        )
    support = normal_density_entry_support(
        keys=[(0,)],
        interaction_support=dense_hint,
        ndof=2,
        local_key=(0,),
        allow_empty=True,
    )
    assert support is not None
    np.testing.assert_allclose(
        tb_to_rparams(dense_result.density_matrix_result.density_matrix, support=support),
        tb_to_rparams(sparse_result.density_matrix_result.density_matrix, support=support),
        atol=5e-4,
    )


def test_zero_dimensional_normal_rational_matches_direct_reference():
    tb_zero_dim = {tuple(): np.diag([-1.0, 1.0]).astype(complex)}
    keys = [tuple()]
    matrix_function = RationalFOE(initial_poles=4, max_poles=256)

    reference_at_mu = density_matrix_at_mu(
        tb_zero_dim,
        mu=0.1,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            matrix_function=DirectDiagonalization(),
        ),
    )
    result_at_mu = density_matrix_at_mu(
        tb_zero_dim,
        mu=0.1,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            matrix_function=matrix_function,
        ),
    )

    assert (
        max_density_error(result_at_mu.density_matrix, reference_at_mu.density_matrix)
        <= 1e-2
    )

    reference_fixed = density_matrix(
        tb_zero_dim,
        filling=0.9,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
    )
    result_fixed = density_matrix(
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

    assert max_density_error(result_fixed.density_matrix, reference_fixed.density_matrix) <= 1e-2
    assert abs(result_fixed.filling - 0.9) <= 1e-2


def test_zero_dimensional_normal_rational_minimax_matches_direct_reference_at_mu():
    tb_zero_dim = {tuple(): np.diag([-1.0, 1.0]).astype(complex)}
    keys = [tuple()]
    matrix_function = RationalFOE(
        initial_poles=4,
        max_poles=16,
        rational_scheme="minimax",
    )

    reference = density_matrix_at_mu(
        tb_zero_dim,
        mu=0.1,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            matrix_function=DirectDiagonalization(),
        ),
    )
    result = density_matrix_at_mu(
        tb_zero_dim,
        mu=0.1,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            matrix_function=matrix_function,
        ),
    )

    assert max_density_error(result.density_matrix, reference.density_matrix) <= 1e-2


@pytest.mark.parametrize("derivative", [False, True], ids=["occupation", "derivative"])
@pytest.mark.parametrize("order", [8, 31], ids=["order-8", "order-31"])
def test_shared_chebyshev_coefficients_match_loop_formula(derivative, order):
    coeffs = matrix_function_common.chebyshev_foe_coefficients(
        order,
        center=0.17,
        scale=1.83,
        kT=0.21,
        mu=-0.34,
        oversampling=5,
        derivative=derivative,
    )
    reference = _loop_chebyshev_coefficients(
        order,
        center=0.17,
        scale=1.83,
        kT=0.21,
        mu=-0.34,
        oversampling=5,
        derivative=derivative,
    )

    np.testing.assert_allclose(coeffs, reference, atol=1e-12, rtol=1e-12)


def test_prepared_normal_chebyshev_node_reuses_only_trace_moments():
    matrix = np.array([[0.2, -1.0], [-1.0, -0.1]], dtype=complex)
    node = prepared_normal.PreparedNormalChebyshevNode(
        matrix,
        kT=0.15,
        options=ChebyshevFOE(initial_order=8, max_order=256),
        charge_tolerance=1e-2,
    )

    assert not hasattr(node, "_basis_terms")
    assert not hasattr(node, "_density_cache")
    assert len(node._trace_moments) == 1

    charge, derivative = node.charge_and_derivative(0.0)
    cached_order = node._charge_cache[0.0][2]
    trace_count = len(node._trace_moments)

    assert np.isfinite(charge)
    assert np.isfinite(derivative)
    assert trace_count == cached_order + 1
    assert node._prepared_order == cached_order
    assert node._tail_prev.shape == matrix.shape
    assert len(node._charge_cache[0.0]) == 3

    density = node.density_from_charge_order(0.0)

    assert density.shape == matrix.shape
    assert len(node._trace_moments) == trace_count
    assert node._prepared_order == cached_order


def test_prepared_normal_chebyshev_node_reuses_cached_coefficients(monkeypatch):
    matrix = np.array([[0.2, -1.0], [-1.0, -0.1]], dtype=complex)
    calls = 0
    original = prepared_normal.chebyshev_foe_coefficients

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(prepared_normal, "chebyshev_foe_coefficients", wrapped)
    node = prepared_normal.PreparedNormalChebyshevNode(
        matrix,
        kT=0.15,
        options=ChebyshevFOE(initial_order=8, max_order=256),
        charge_tolerance=1e-2,
    )

    node.charge_and_derivative(0.0)
    first_call_count = calls
    node.charge_and_derivative(0.0)
    node.density_from_charge_order(0.0)

    assert first_call_count > 0
    assert calls == first_call_count


def test_prepared_normal_chebyshev_hutchinson_charge_is_reproducible_and_close_to_exact():
    matrix = np.array(
        [
            [0.2, -1.0, 0.0, 0.0],
            [-1.0, -0.1, -0.4, 0.0],
            [0.0, -0.4, 0.3, -0.7],
            [0.0, 0.0, -0.7, -0.2],
        ],
        dtype=complex,
    )
    exact_node = prepared_normal.PreparedNormalChebyshevNode(
        matrix,
        kT=0.2,
        options=ChebyshevFOE(initial_order=8, max_order=256),
        charge_tolerance=1e-4,
    )
    hutch_options = ChebyshevFOE(
        initial_order=8,
        max_order=256,
        trace_estimator="hutchinson",
        trace_probes=1024,
        trace_seed=7,
    )
    hutch_node_a = prepared_normal.PreparedNormalChebyshevNode(
        matrix,
        kT=0.2,
        options=hutch_options,
        charge_tolerance=1e-4,
    )
    hutch_node_b = prepared_normal.PreparedNormalChebyshevNode(
        matrix,
        kT=0.2,
        options=hutch_options,
        charge_tolerance=1e-4,
    )

    exact_charge, exact_derivative = exact_node.charge_and_derivative(0.1)
    hutch_charge_a, hutch_derivative_a = hutch_node_a.charge_and_derivative(0.1)
    hutch_charge_b, hutch_derivative_b = hutch_node_b.charge_and_derivative(0.1)

    assert hutch_charge_a == pytest.approx(hutch_charge_b, abs=1e-12)
    assert hutch_derivative_a == pytest.approx(hutch_derivative_b, abs=1e-12)
    assert abs(hutch_charge_a - exact_charge) <= 6e-2
    assert abs(hutch_derivative_a - exact_derivative) <= 5e-2


def test_prepared_rational_hutchinson_charge_is_reproducible_and_close_to_exact():
    matrix = np.array(
        [
            [0.2, -1.0, 0.0, 0.0],
            [-1.0, -0.1, -0.4, 0.0],
            [0.0, -0.4, 0.3, -0.7],
            [0.0, 0.0, -0.7, -0.2],
        ],
        dtype=complex,
    )
    exact_node = rational_matrix_functions.PreparedRationalNode(
        matrix,
        kT=0.2,
        q_diag=np.ones(matrix.shape[0], dtype=float),
        options=RationalFOE(initial_poles=4, max_poles=128),
        charge_tolerance=1e-4,
    )
    hutch_options = RationalFOE(
        initial_poles=4,
        max_poles=128,
        trace_estimator="hutchinson",
        trace_probes=1024,
        trace_seed=11,
    )
    hutch_node_a = rational_matrix_functions.PreparedRationalNode(
        matrix,
        kT=0.2,
        q_diag=np.ones(matrix.shape[0], dtype=float),
        options=hutch_options,
        charge_tolerance=1e-4,
    )
    hutch_node_b = rational_matrix_functions.PreparedRationalNode(
        matrix,
        kT=0.2,
        q_diag=np.ones(matrix.shape[0], dtype=float),
        options=hutch_options,
        charge_tolerance=1e-4,
    )

    exact_charge, exact_derivative = exact_node.charge_and_derivative(0.1)
    hutch_charge_a, hutch_derivative_a = hutch_node_a.charge_and_derivative(0.1)
    hutch_charge_b, hutch_derivative_b = hutch_node_b.charge_and_derivative(0.1)

    assert hutch_charge_a == pytest.approx(hutch_charge_b, abs=1e-12)
    assert hutch_derivative_a == pytest.approx(hutch_derivative_b, abs=1e-12)
    assert abs(hutch_charge_a - exact_charge) <= 5e-2
    assert abs(hutch_derivative_a - exact_derivative) <= 5e-2


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
    exact = np.asarray(fermi_dirac(probe, 0.2, 0.0), dtype=complex)
    assert np.max(np.abs(approximation - exact)) <= 1e-2


def test_sparse_aaa_tail_shortcut_returns_constant_zero_pole_terms():
    terms, _builder = rational_matrix_functions._aaa_terms_for_interval(
        512,
        lower=5.0,
        upper=7.0,
        kT=0.2,
        initial_poles=4,
        scalar_tolerance=1e-2,
    )
    assert terms.shifts.size == 0
    assert terms.residues.size == 0
    assert terms.constant == pytest.approx(0.0)


def test_vectorized_barycentric_evaluation_matches_scalar_reference():
    support_x = np.array([-1.5, -0.2, 0.7, 1.8], dtype=float)
    support_y = np.array([0.9, 0.6, 0.25, 0.05], dtype=complex)
    weights = np.array([1.0, -0.4, 0.7, -0.2], dtype=complex)
    probe = np.linspace(-2.0, 2.0, 401, dtype=float)
    vectorized = rational_matrix_functions._barycentric_evaluate(
        probe,
        support_x,
        support_y,
        weights,
    )

    scalar = np.empty_like(vectorized)
    for index, point in enumerate(probe):
        diff = point - support_x
        exact = np.abs(diff) <= 32.0 * np.finfo(float).eps * max(
            1.0,
            float(np.max(np.abs(support_x))),
            abs(point),
        )
        if np.any(exact):
            scalar[index] = support_y[int(np.flatnonzero(exact)[0])]
            continue
        scaled = weights / diff
        scalar[index] = np.dot(scaled, support_y) / np.sum(scaled)

    np.testing.assert_allclose(vectorized, scalar, atol=1e-12, rtol=1e-12)


def test_sparse_aaa_arrowhead_poles_reproduce_barycentric_values():
    training_grid = rational_matrix_functions._aaa_sample_grid(-2.0, 2.5, count=256, kT=0.2)
    target = np.asarray(fermi_dirac(training_grid, 0.2, 0.0), dtype=complex)
    builder = rational_matrix_functions._initialize_aaa_builder(
        training_grid,
        target,
        training_grid,
        target,
    )
    while len(builder.support_indices) < 12:
        builder = rational_matrix_functions._advance_aaa_builder(builder)

    constant, poles, residues = rational_matrix_functions._extract_aaa_poles_arrowhead(
        builder.support_x,
        builder.support_y,
        builder.weights,
    )
    constant, shifts, residues = rational_matrix_functions._canonicalize_conjugate_terms(
        constant,
        poles,
        residues,
    )
    barycentric = rational_matrix_functions._barycentric_evaluate(
        training_grid,
        builder.support_x,
        builder.support_y,
        builder.weights,
    )
    pole_values = rational_matrix_functions._evaluate_canonical_rational(
        training_grid,
        constant=constant,
        shifts=shifts,
        residues=residues,
    )
    assert np.max(np.abs(pole_values - barycentric)) <= 1e-3


def test_sparse_aaa_canonicalization_preserves_real_axis_values():
    constant = 0.5 + 0.0j
    poles = np.array([0.3 + 1.2j, 0.3 - 1.2j, -0.7 + 0.0j], dtype=complex)
    residues = np.array([0.2 - 0.05j, 0.2 + 0.05j, 0.12 + 0.0j], dtype=complex)
    canonical_constant, canonical_poles, canonical_residues = (
        rational_matrix_functions._canonicalize_conjugate_terms(
            constant,
            poles,
            residues,
        )
    )
    probe = np.linspace(-3.0, 3.0, 2001, dtype=float)
    original = (
        constant
        + residues[0] / (probe - poles[0])
        + residues[1] / (probe - poles[1])
        + residues[2] / (probe - poles[2])
    )
    canonical = rational_matrix_functions._evaluate_canonical_rational(
        probe,
        constant=canonical_constant,
        shifts=canonical_poles,
        residues=canonical_residues,
    )
    assert canonical_poles.size < poles.size
    np.testing.assert_allclose(canonical, original, atol=1e-10, rtol=1e-10)


def test_sparse_rational_aaa_matches_direct_reference_at_mu():
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix_at_mu(
        spinful_chain(),
        mu=0.05,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            matrix_function=DirectDiagonalization(),
        ),
    )
    result = density_matrix_at_mu(
        sparse_tb,
        mu=0.05,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=20,
            matrix_function=RationalFOE(
                initial_poles=4,
                max_poles=512,
                rational_scheme="aaa",
            ),
        ),
    )

    assert max_density_error(result.density_matrix, reference.density_matrix) <= 1e-2


def test_sparse_rational_aaa_matches_direct_reference_at_fixed_filling():
    sparse_tb = _sparse_tb(spinful_chain())
    keys = [(0,), (1,), (-1,)]
    reference = density_matrix(
        spinful_chain(),
        filling=1.0,
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
        filling=1.0,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=20,
            matrix_function=RationalFOE(
                initial_poles=4,
                max_poles=512,
                rational_scheme="aaa",
            ),
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert abs(result.mu - reference.mu) <= 1e-2
    assert abs(result.filling - reference.filling) <= 1e-2
    assert max_density_error(result.density_matrix, reference.density_matrix) <= 1e-2


def test_sparse_aaa_interval_cache_reuses_nested_interval():
    pytest.importorskip("mumps")
    sparse = pytest.importorskip("scipy.sparse")
    matrix = sparse.csr_matrix(np.array([[0.2, 0.1], [0.1, -0.3]], dtype=complex))
    support = normal_density_entry_support(
        keys=[(0,)],
        interaction_support={(0,): np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)},
        ndof=2,
        local_key=(0,),
    )
    assert support is not None
    node = rational_matrix_functions.PreparedMumpsRationalNode(
        matrix,
        kT=0.2,
        q_diag=np.ones(2, dtype=float),
        options=RationalFOE(initial_poles=4, max_poles=512, rational_scheme="aaa"),
        charge_tolerance=1e-2,
        density_support=support,
        density_tolerance=1e-2,
    )
    first = node._sparse_terms(0.0, pole_count=512)
    assert len(node._aaa_interval_cache) == 1
    cached_interval = node._aaa_interval_cache[0]
    reused = node._aaa_cached_terms_for_interval(
        lower=cached_interval.lower + 0.05,
        upper=cached_interval.upper - 0.05,
        pole_cap=512,
    )
    assert reused is not None
    assert reused.support_count == first.support_count


def test_density_support_builders_select_only_interaction_entries():
    interaction = {
        (0,): np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex),
        (1,): np.array([[0.0, 3.0], [0.0, 0.0]], dtype=complex),
    }
    normal_support = normal_density_entry_support(
        keys=[(0,), (1,)],
        interaction_support=interaction,
        ndof=2,
        local_key=(0,),
    )
    assert normal_support is not None
    assert tuple(normal_support.selected_columns.tolist()) == (0, 1)
    assert normal_support.output_size == 3
    np.testing.assert_array_equal(normal_support.row_indices[0], np.array([0, 1]))
    np.testing.assert_array_equal(normal_support.col_indices[0], np.array([0, 1]))
    np.testing.assert_array_equal(normal_support.row_indices[1], np.array([0]))
    np.testing.assert_array_equal(normal_support.col_indices[1], np.array([1]))

    bdg_support = bdg_density_entry_support(
        keys=[(0,), (1,)],
        interaction_support=interaction,
        ndof=2,
        local_key=(0,),
    )
    assert tuple(bdg_support.selected_columns.tolist()) == (0, 1, 2, 3)
    assert bdg_support.output_size == 6
    np.testing.assert_array_equal(bdg_support.row_indices[0], np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(bdg_support.col_indices[0], np.array([0, 2, 1, 3]))
    np.testing.assert_array_equal(bdg_support.row_indices[1], np.array([0, 0]))
    np.testing.assert_array_equal(bdg_support.col_indices[1], np.array([1, 3]))


def test_normal_density_support_tracks_only_structurally_touched_diagonals():
    support = normal_density_entry_support(
        keys=[(0,), (1,), (-1,)],
        interaction_support={
            (0,): np.zeros((3, 3), dtype=complex),
            (1,): np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=complex,
            ),
            (-1,): np.array(
                [
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=complex,
            ),
        },
        ndof=3,
        local_key=(0,),
        allow_empty=True,
    )
    assert support is not None
    np.testing.assert_array_equal(support.row_indices[0], np.array([0, 1]))
    np.testing.assert_array_equal(support.col_indices[0], np.array([0, 1]))


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


@requires_prepared_payloads
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
            matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
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
            matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
            workspace_precision=64,
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert abs(low_precision.mu - high_precision.mu) <= 5e-3
    assert max_density_error(
        low_precision.density_matrix,
        high_precision.density_matrix,
    ) <= 2e-2


@requires_prepared_payloads
def test_solver_density_result_zeroes_entries_outside_interaction_support():
    sparse = pytest.importorskip("scipy.sparse")
    h0 = {(0,): sparse.csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=complex))}
    h_int = {(0,): sparse.csr_matrix(np.diag([1.0, 1.0]).astype(complex))}
    model = Model(h0, h_int, filling=1.0, kT=0.15)
    result = solver(
        model,
        {(0,): sparse.csr_matrix(np.zeros((2, 2), dtype=complex))},
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=40,
            matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
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
            matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
        ),
        scf=LinearMixing(max_iterations=1, alpha=0.5),
        scf_tol=1e-8,
        filling_tol=1e-2,
    )
    onsite_block = result.density_matrix_result.density_matrix[(0,)]
    assert not np.allclose(np.diag(np.diag(onsite_block)), onsite_block, atol=1e-12)


@requires_prepared_payloads
def test_fixed_filling_chebyshev_density_pass_uses_frozen_charge_mesh(monkeypatch):
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
    result = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=60,
            matrix_function=ChebyshevFOE(initial_order=8, max_order=256),
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


@requires_prepared_payloads
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

    class _FakePreparedNode:
        size = 2

        def charge_and_derivative(self, mu: float) -> tuple[float, float]:
            return 1.0 + float(mu), 1.0

        def density_from_charge_order(self, mu: float) -> np.ndarray:
            del mu
            return np.eye(2, dtype=complex)

        def density_columns_from_charge_order(self, mu: float, basis: np.ndarray) -> np.ndarray:
            del mu
            return np.asarray(basis, dtype=complex)

    def fake_builder(*, matrix_function, matrix_from_payload, kT, charge_tolerance, workspace_dtype, q_diag, trace_weights_diag):
        del matrix_function, matrix_from_payload, kT, charge_tolerance, workspace_dtype, q_diag, trace_weights_diag

        def builder(points: np.ndarray, payload: np.ndarray):
            del points
            return [_FakePreparedNode() for _ in range(len(payload))]

        return builder

    monkeypatch.setattr(
        normal_quadrature_backend,
        "prepared_rational_payload_builder",
        fake_builder,
    )
    result = density_matrix(
        local_two_band_2d(),
        filling=1.0,
        kT=0.2,
        keys=[(0, 0), (1, 0), (0, 1)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-1,
            max_refinements=20,
            matrix_function=RationalFOE(initial_poles=4, max_poles=64),
        ),
        filling_tol=1e-1,
        mu_tol=1e-8,
    )

    assert len(calls) >= 2
    assert calls[-1]["max_subdivisions"] == 0
    assert calls[-1]["status"] in {"converged", "max_subdivisions"}
    assert calls[-1]["n_kernel_evals"] == 0
    assert calls[-1]["subdivisions"] == 0
    assert result.info.n_kernel_evals == result.info.unique_evals
