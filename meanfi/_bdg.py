from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from meanfi._finite_temp import (
    charge_integral_tolerance,
    expand_mu_bracket,
    fermi_dirac,
    integration_bounds,
    quadrature_prefactor,
    run_integrator,
    solve_mu,
    split_density_result,
)
from meanfi._info import AdaptiveQuadratureInfo, DensityMatrixResult, FixedFillingInfo, SCFInfo
from meanfi._validation import _matrix_allclose
from meanfi.bdg import BdGMatrixFunction, ChebyshevFOE, ExactDiagonalization
from meanfi.integration import AdaptiveQuadrature, IntegrationMethod
from meanfi.mf import meanfield as normal_meanfield
from meanfi.params.rparams import bdg_tb_to_rparams
from meanfi.scf import LinearMixing
from meanfi.tb.tb import _tb_type


@dataclass(frozen=True)
class _BlockResult:
    block: np.ndarray
    derivative_block: np.ndarray | None
    error: float
    order: int | None


def _is_sparse_like(matrix: Any) -> bool:
    return hasattr(matrix, "toarray") and hasattr(matrix, "tocsr")


def _sparse_module():
    try:
        import scipy.sparse as sparse
    except ImportError as exc:  # pragma: no cover - depends on optional scipy
        raise ImportError("Sparse BdG inputs require scipy to be installed") from exc
    return sparse


def _to_dense(matrix: Any) -> np.ndarray:
    if _is_sparse_like(matrix):
        return np.asarray(matrix.toarray(), dtype=complex)
    return np.asarray(matrix, dtype=complex)


def _as_sparse(matrix: Any):
    sparse = _sparse_module()
    if _is_sparse_like(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix, dtype=complex))


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("BdG tight-binding values must be matrices")
    return int(shape[0]), int(shape[1])


def _transpose(matrix: Any):
    return matrix.T


def _conjugate_transpose(matrix: Any):
    return matrix.conj().T


def _elementwise_product(lhs: Any, rhs: Any):
    if _is_sparse_like(lhs):
        return lhs.multiply(np.asarray(rhs, dtype=complex)).tocsr()
    if _is_sparse_like(rhs):
        return rhs.multiply(np.asarray(lhs, dtype=complex)).tocsr()
    return np.asarray(lhs, dtype=complex) * np.asarray(rhs, dtype=complex)


def _block_diag(top: Any, bottom: Any):
    if _is_sparse_like(top) or _is_sparse_like(bottom):
        sparse = _sparse_module()
        return sparse.bmat(
            [[_as_sparse(top), None], [None, _as_sparse(bottom)]],
            format="csr",
        )

    top = np.asarray(top, dtype=complex)
    bottom = np.asarray(bottom, dtype=complex)
    zero_top = np.zeros((top.shape[0], bottom.shape[1]), dtype=complex)
    zero_bottom = np.zeros((bottom.shape[0], top.shape[1]), dtype=complex)
    return np.block([[top, zero_top], [zero_bottom, bottom]])


def electron_to_bdg_tb(h: _tb_type, ndof: int) -> _tb_type:
    """Embed an electron-space tight-binding Hamiltonian into electron-first BdG space."""

    zero = np.zeros((ndof, ndof), dtype=complex)
    keys = set(h)
    keys.update(tuple(-np.asarray(key, dtype=int)) for key in h)
    bdg = {}
    for key in keys:
        opposite = tuple(-np.asarray(key, dtype=int))
        top = h.get(key, zero)
        bottom = -_transpose(h.get(opposite, zero))
        bdg[key] = _block_diag(top, bottom)
    return bdg


def charge_diagonal(ndof: int) -> np.ndarray:
    """Return the electron-first Nambu charge diagonal."""

    return np.concatenate([np.ones(ndof), -np.ones(ndof)])


def _basis_block(size: int, columns: Sequence[int]) -> np.ndarray:
    block = np.zeros((size, len(columns)), dtype=complex)
    block[np.asarray(columns, dtype=int), np.arange(len(columns))] = 1.0
    return block


def _matrix_action(matrix: Any, block: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ block, dtype=complex)


def _shift_by_mu(matrix: Any, mu: float, q_diag: np.ndarray):
    if _is_sparse_like(matrix):
        sparse = _sparse_module()
        return matrix - float(mu) * sparse.diags(q_diag, format="csr")
    return np.asarray(matrix, dtype=complex) - float(mu) * np.diag(q_diag)


def _matrix_bound(matrix: Any) -> float:
    if _is_sparse_like(matrix):
        row_sums = np.asarray(abs(matrix).sum(axis=1)).ravel()
        return float(np.max(row_sums)) if row_sums.size else 0.0
    array = np.asarray(matrix)
    if array.size == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(array), axis=1)))


def _mu_bracket_for_bdg(hamiltonian: _tb_type, kT: float) -> tuple[float, float]:
    bound = sum(_matrix_bound(matrix) for matrix in hamiltonian.values())
    padding = max(1.0, 10.0 * kT)
    return -float(bound + padding), float(bound + padding)


def _gershgorin_bounds(matrix: Any) -> tuple[float, float]:
    if _is_sparse_like(matrix):
        diagonal = np.asarray(matrix.diagonal(), dtype=complex)
        row_sums = np.asarray(abs(matrix).sum(axis=1)).ravel()
    else:
        array = np.asarray(matrix, dtype=complex)
        diagonal = np.diag(array)
        row_sums = np.sum(np.abs(array), axis=1)

    radius = np.maximum(row_sums - np.abs(diagonal), 0.0)
    center = diagonal.real
    return float(np.min(center - radius)), float(np.max(center + radius))


def _chebyshev_coefficients(
    order: int,
    *,
    center: float,
    scale: float,
    kT: float,
    oversampling: int,
) -> np.ndarray:
    n_nodes = max(64, int(oversampling) * (order + 1))
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    values = fermi_dirac(scale * np.cos(theta) + center, kT, 0.0)
    coeffs = np.empty(order + 1, dtype=float)
    coeffs[0] = np.mean(values)
    for mode in range(1, order + 1):
        coeffs[mode] = 2.0 * np.mean(values * np.cos(mode * theta))
    return coeffs


def _exact_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
) -> _BlockResult:
    array = _to_dense(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    occupation = fermi_dirac(eigenvalues, kT, 0.0)
    projected_block = eigenvectors.conj().T @ block
    density_block = eigenvectors @ (occupation[:, np.newaxis] * projected_block)

    derivative_block = None
    if derivative:
        fprime = -occupation * (1.0 - occupation) / kT
        delta = eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]
        numerator = occupation[:, np.newaxis] - occupation[np.newaxis, :]
        loewner = np.empty_like(delta, dtype=float)
        separated = np.abs(delta) > 1e-12
        loewner[separated] = numerator[separated] / delta[separated]
        loewner[~separated] = np.broadcast_to(
            fprime[:, np.newaxis], delta.shape
        )[~separated]

        d_h = -(q_diag[:, np.newaxis] * eigenvectors)
        projected_dh = eigenvectors.conj().T @ d_h
        derivative_block = eigenvectors @ ((loewner * projected_dh) @ projected_block)

    return _BlockResult(
        block=density_block,
        derivative_block=derivative_block,
        error=0.0,
        order=None,
    )


def _apply_rescaled(matrix: Any, block: np.ndarray, *, center: float, scale: float) -> np.ndarray:
    return (_matrix_action(matrix, block) - center * block) / scale


def _chebyshev_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    options: ChebyshevFOE,
) -> _BlockResult:
    lower, upper = _gershgorin_bounds(matrix)
    width = max(upper - lower, 1e-12)
    scale = 0.5 * width * (1.0 + options.spectral_padding)
    center = 0.5 * (upper + lower)

    accepted_block = None
    accepted_derivative = None
    accepted_error = float("inf")
    accepted_order = None

    order_half = int(options.initial_order)
    while 2 * order_half <= int(options.max_order):
        order = 2 * order_half
        coeffs = _chebyshev_coefficients(
            order,
            center=center,
            scale=scale,
            kT=kT,
            oversampling=options.coefficient_oversampling,
        )

        v_prev = np.array(block, copy=True)
        v_curr = _apply_rescaled(matrix, block, center=center, scale=scale)
        y_half = coeffs[0] * v_prev
        y_full = coeffs[0] * v_prev

        if derivative:
            s_prev = np.zeros_like(block)
            s_curr = -(q_diag[:, np.newaxis] * block) / scale
            dy_half = coeffs[0] * s_prev
            dy_full = coeffs[0] * s_prev
        else:
            s_prev = s_curr = None
            dy_half = dy_full = None

        if order_half >= 1:
            y_half = y_half + coeffs[1] * v_curr
        y_full = y_full + coeffs[1] * v_curr
        if derivative:
            if order_half >= 1:
                dy_half = dy_half + coeffs[1] * s_curr
            dy_full = dy_full + coeffs[1] * s_curr

        for mode in range(1, order):
            v_next = 2.0 * _apply_rescaled(
                matrix,
                v_curr,
                center=center,
                scale=scale,
            ) - v_prev
            if derivative:
                d_x_v = -(q_diag[:, np.newaxis] * v_curr) / scale
                s_next = (
                    2.0
                    * _apply_rescaled(
                        matrix,
                        s_curr,
                        center=center,
                        scale=scale,
                    )
                    + 2.0 * d_x_v
                    - s_prev
                )

            term_index = mode + 1
            if term_index <= order_half:
                y_half = y_half + coeffs[term_index] * v_next
                if derivative:
                    dy_half = dy_half + coeffs[term_index] * s_next
            y_full = y_full + coeffs[term_index] * v_next
            if derivative:
                dy_full = dy_full + coeffs[term_index] * s_next

            v_prev, v_curr = v_curr, v_next
            if derivative:
                s_prev, s_curr = s_curr, s_next

        accepted_error = float(np.max(np.abs(y_full - y_half)))
        accepted_block = y_full
        accepted_derivative = dy_full
        accepted_order = order
        if accepted_error <= tolerance:
            break
        order_half = order

    if accepted_error > tolerance:
        raise ValueError("Chebyshev FOE did not converge within max_order")

    return _BlockResult(
        block=accepted_block,
        derivative_block=accepted_derivative,
        error=accepted_error,
        order=accepted_order,
    )


def _density_block(
    matrix_function: BdGMatrixFunction,
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
) -> _BlockResult:
    if isinstance(matrix_function, ExactDiagonalization):
        return _exact_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
        )
    if isinstance(matrix_function, ChebyshevFOE):
        return _chebyshev_density_block(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            derivative=derivative,
            tolerance=tolerance,
            options=matrix_function,
        )
    raise TypeError("matrix_function must be a BdGMatrixFunction instance")


def _tb_k_matrix(hamiltonian: _tb_type, point: np.ndarray):
    accumulator = None
    for key, matrix in hamiltonian.items():
        phase = np.exp(-1j * np.dot(point, np.asarray(key, dtype=float)))
        term = matrix * phase
        accumulator = term if accumulator is None else accumulator + term
    if accumulator is None:
        raise ValueError("BdG Hamiltonian cannot be empty")
    return accumulator


def _dummy_payload(points: np.ndarray) -> np.ndarray:
    return np.zeros((points.shape[0], 1), dtype=float)


def _build_bdg_integrator(*, ndim: int, evaluator, integration: AdaptiveQuadrature):
    try:
        from stateful_quadrature import StatefulIntegrator
    except ImportError as exc:  # pragma: no cover - dependency is declared by meanfi
        raise ImportError(
            "BdG adaptive quadrature requires stateful_quadrature"
        ) from exc

    a, b = integration_bounds(ndim)
    return StatefulIntegrator(
        a=a,
        b=b,
        kernel=_dummy_payload,
        evaluator=evaluator,
        rule=integration.rule,
        batch_size=integration.batch_size,
    )


def _local_filling(block: np.ndarray, indices: Sequence[int], weights: np.ndarray) -> float:
    values = block[np.asarray(indices, dtype=int), np.arange(len(indices))]
    return float(np.real(np.sum(weights * values)))


def _charge_evaluator(
    *,
    hamiltonian: _tb_type,
    ndim: int,
    kT: float,
    q_diag: np.ndarray,
    matrix_function: BdGMatrixFunction,
    tolerance: float,
    filling_indices: Sequence[int],
    filling_weights: np.ndarray,
):
    prefactor = quadrature_prefactor(ndim)
    size = q_diag.size
    filling_block = _basis_block(size, filling_indices)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        del payload
        values = np.empty((points.shape[0], 2), dtype=float)
        for index, point in enumerate(points):
            matrix = _shift_by_mu(_tb_k_matrix(hamiltonian, point), mu, q_diag)
            result = _density_block(
                matrix_function,
                matrix,
                filling_block,
                kT=kT,
                q_diag=q_diag,
                derivative=True,
                tolerance=tolerance,
            )
            derivative_block = result.derivative_block
            derivative = (
                0.0
                if derivative_block is None
                else _local_filling(derivative_block, filling_indices, filling_weights)
            )
            values[index, 0] = _local_filling(
                result.block,
                filling_indices,
                filling_weights,
            )
            values[index, 1] = derivative
        return prefactor * values

    return evaluator


def _density_evaluator(
    *,
    hamiltonian: _tb_type,
    ndim: int,
    kT: float,
    q_diag: np.ndarray,
    matrix_function: BdGMatrixFunction,
    tolerance: float,
    keys: list[tuple[int, ...]],
):
    prefactor = quadrature_prefactor(ndim)
    keys_array = np.asarray(keys, dtype=float)
    size = q_diag.size
    density_block = np.eye(size, dtype=complex)

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        del payload
        values = np.empty((points.shape[0], size * size * len(keys)), dtype=complex)
        for index, point in enumerate(points):
            matrix = _shift_by_mu(_tb_k_matrix(hamiltonian, point), mu, q_diag)
            result = _density_block(
                matrix_function,
                matrix,
                density_block,
                kT=kT,
                q_diag=q_diag,
                derivative=False,
                tolerance=tolerance,
            )
            phase = np.exp(1j * np.dot(point, keys_array.T))
            values[index] = (
                result.block[..., np.newaxis] * phase[np.newaxis, np.newaxis, :]
            ).reshape(-1)
        return prefactor * values

    return evaluator


def _split_charge(estimate: np.ndarray, error: np.ndarray) -> tuple[float, float, float]:
    estimate = np.asarray(estimate)
    error = np.asarray(error)
    return float(np.real(estimate[0])), float(abs(error[0])), float(np.real(estimate[1]))


def _adaptive_info(raw_info: FixedFillingInfo) -> AdaptiveQuadratureInfo:
    return AdaptiveQuadratureInfo(
        n_kernel_evals=int(raw_info.n_kernel_evals),
        unique_evals=int(raw_info.unique_evals),
        n_evaluator_evals=int(raw_info.n_evaluator_evals),
        n_cached_nodes=int(raw_info.n_cached_nodes),
        n_leaves=int(raw_info.n_leaves),
        n_leaf_nodes=int(raw_info.n_leaf_nodes),
        refinements=int(raw_info.subdivisions),
        error_estimate_available=bool(raw_info.error_estimate_available),
        root_iterations=int(raw_info.root_iterations),
        charge_integration_calls=int(raw_info.charge_integration_calls),
        density_integration_calls=int(raw_info.density_integration_calls),
    )


def _wrap_result(
    *,
    density_matrix: _tb_type,
    density_matrix_error: _tb_type,
    raw_info: FixedFillingInfo,
    target_filling: float,
    integration: AdaptiveQuadrature,
) -> DensityMatrixResult:
    return DensityMatrixResult(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        mu=float(raw_info.mu),
        filling=float(raw_info.charge),
        target_filling=float(target_filling),
        filling_residual=abs(float(raw_info.charge) - float(target_filling)),
        integration=integration,
        info=_adaptive_info(raw_info),
    )


def _effective_filling_tol(
    *,
    filling_tol: float | None,
    density_matrix_tol: float,
    filling_weights: np.ndarray,
) -> float:
    if filling_tol is not None:
        if filling_tol <= 0:
            raise ValueError("filling_tol must be positive when provided")
        return float(filling_tol)
    return float(np.sum(np.abs(filling_weights)) * density_matrix_tol)


def _matrix_function(integration: AdaptiveQuadrature) -> BdGMatrixFunction:
    matrix_function = getattr(integration, "matrix_function", None)
    if matrix_function is None:
        return ExactDiagonalization()
    if not isinstance(matrix_function, BdGMatrixFunction):
        raise TypeError("AdaptiveQuadrature.matrix_function must be a BdGMatrixFunction")
    return matrix_function


def _solve_bdg_zero_dim(
    *,
    hamiltonian: _tb_type,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: AdaptiveQuadrature,
    filling_tol: float,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
    q_diag: np.ndarray,
    matrix_function: BdGMatrixFunction,
    filling_indices: Sequence[int],
    filling_weights: np.ndarray,
) -> DensityMatrixResult:
    matrix = hamiltonian[tuple()]
    filling_block = _basis_block(q_diag.size, filling_indices)

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        result = _density_block(
            matrix_function,
            _shift_by_mu(matrix, mu, q_diag),
            filling_block,
            kT=kT,
            q_diag=q_diag,
            derivative=True,
            tolerance=integration.density_matrix_tol,
        )
        derivative_block = result.derivative_block
        derivative = (
            0.0
            if derivative_block is None
            else _local_filling(derivative_block, filling_indices, filling_weights)
        )
        return _local_filling(result.block, filling_indices, filling_weights), 0.0, derivative

    lower, upper = _mu_bracket_for_bdg(hamiltonian, kT)
    lower, upper = expand_mu_bracket(
        evaluate_charge,
        filling=filling,
        lower=lower,
        upper=upper,
    )
    mu, resolved_filling, charge_error, derivative, iteration = solve_mu(
        evaluate_charge,
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=filling_tol,
        mu_xtol=mu_tol,
        max_mu_iterations=max_mu_iterations,
    )

    density_block = np.eye(q_diag.size, dtype=complex)
    density = _density_block(
        matrix_function,
        _shift_by_mu(matrix, mu, q_diag),
        density_block,
        kT=kT,
        q_diag=q_diag,
        derivative=False,
        tolerance=integration.density_matrix_tol,
    ).block
    density_matrix = {key: density if key == tuple() else np.zeros_like(density) for key in keys}
    density_matrix_error = {key: np.zeros_like(density) for key in keys}
    raw_info = FixedFillingInfo(
        mu=mu,
        charge=resolved_filling,
        charge_error=charge_error,
        dcharge_dmu=derivative,
        root_iterations=iteration,
        charge_integration_calls=iteration,
        density_integration_calls=1,
        charge_n_kernel_evals=1,
        density_n_kernel_evals=1,
        n_kernel_evals=1,
        unique_evals=1,
        charge_n_evaluator_evals=iteration,
        density_n_evaluator_evals=1,
        n_evaluator_evals=iteration + 1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
        charge_integral_atol=filling_tol,
        density_atol=integration.density_matrix_tol,
        density_rtol=0.0,
        error_estimate_available=True,
    )
    return _wrap_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        target_filling=filling,
        integration=integration,
    )


def solve_bdg_density_fixed_filling(
    model,
    meanfield: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    mu_guess: float,
) -> DensityMatrixResult:
    if not isinstance(integration, AdaptiveQuadrature):
        raise NotImplementedError("BdG density currently requires AdaptiveQuadrature")
    if model.kT <= 0:
        raise NotImplementedError("BdG density currently requires kT > 0")
    if mu_tol <= 0:
        raise ValueError("mu_tol must be positive")
    if max_mu_iterations is not None and max_mu_iterations <= 0:
        raise ValueError("max_mu_iterations must be positive")

    matrix_function = _matrix_function(integration)
    hamiltonian = model.bdg_hamiltonian_from_meanfield(meanfield)
    q_diag = charge_diagonal(model._ndof)
    filling_indices = tuple(range(model._ndof))
    filling_weights = np.ones(model._ndof, dtype=float)
    resolved_filling_tol = _effective_filling_tol(
        filling_tol=filling_tol,
        density_matrix_tol=integration.density_matrix_tol,
        filling_weights=filling_weights,
    )

    if model._ndim == 0:
        return _solve_bdg_zero_dim(
            hamiltonian=hamiltonian,
            filling=model.filling,
            kT=model.kT,
            keys=keys,
            integration=integration,
            filling_tol=resolved_filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_guess=mu_guess,
            q_diag=q_diag,
            matrix_function=matrix_function,
            filling_indices=filling_indices,
            filling_weights=filling_weights,
        )

    charge_integral_atol, charge_integral_rtol = charge_integral_tolerance(
        resolved_filling_tol
    )
    charge_integration_calls = 0
    charge_kernel_evals = 0
    charge_evaluator_evals = 0
    charge_refinements = 0

    def evaluate_charge(candidate_mu: float) -> tuple[float, float, float]:
        nonlocal charge_integration_calls, charge_kernel_evals, charge_evaluator_evals, charge_refinements
        charge_integrator = _build_bdg_integrator(
            ndim=model._ndim,
            evaluator=_charge_evaluator(
                hamiltonian=hamiltonian,
                ndim=model._ndim,
                kT=model.kT,
                q_diag=q_diag,
                matrix_function=matrix_function,
                tolerance=integration.density_matrix_tol,
                filling_indices=filling_indices,
                filling_weights=filling_weights,
            ),
            integration=integration,
        )
        result = run_integrator(
            charge_integrator,
            candidate_mu,
            atol=charge_integral_atol,
            rtol=charge_integral_rtol,
            max_subdivisions=integration.max_refinements,
            error_message=(
                "BdG adaptive quadrature did not converge while solving for the chemical potential"
            ),
        )
        charge_integration_calls += 1
        charge_kernel_evals += int(result.n_kernel_evals)
        charge_evaluator_evals += int(result.n_evaluator_evals)
        charge_refinements += int(getattr(result, "subdivisions", 0))
        return _split_charge(result.estimate, result.error)

    lower, upper = _mu_bracket_for_bdg(hamiltonian, model.kT)
    lower, upper = expand_mu_bracket(
        evaluate_charge,
        filling=model.filling,
        lower=lower,
        upper=upper,
    )
    mu, resolved_filling, charge_error, derivative, iteration = solve_mu(
        evaluate_charge,
        filling=model.filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=resolved_filling_tol,
        mu_xtol=mu_tol,
        max_mu_iterations=max_mu_iterations,
    )

    density_integrator = _build_bdg_integrator(
        ndim=model._ndim,
        evaluator=_density_evaluator(
            hamiltonian=hamiltonian,
            ndim=model._ndim,
            kT=model.kT,
            q_diag=q_diag,
            matrix_function=matrix_function,
            tolerance=integration.density_matrix_tol,
            keys=keys,
        ),
        integration=integration,
    )
    density_result = run_integrator(
        density_integrator,
        mu,
        atol=integration.density_matrix_tol,
        rtol=0.0,
        max_subdivisions=integration.max_refinements,
        error_message="BdG adaptive quadrature did not converge while evaluating density",
    )
    density_matrix, density_matrix_error = split_density_result(
        density_result.estimate,
        density_result.error,
        2 * model._ndof,
        keys,
    )
    raw_info = FixedFillingInfo(
        mu=mu,
        charge=resolved_filling,
        charge_error=charge_error,
        dcharge_dmu=derivative,
        root_iterations=iteration,
        charge_integration_calls=charge_integration_calls,
        density_integration_calls=1,
        charge_n_kernel_evals=charge_kernel_evals,
        density_n_kernel_evals=int(density_result.n_kernel_evals),
        n_kernel_evals=charge_kernel_evals + int(density_result.n_kernel_evals),
        unique_evals=charge_kernel_evals + int(density_result.n_kernel_evals),
        charge_n_evaluator_evals=charge_evaluator_evals,
        density_n_evaluator_evals=int(density_result.n_evaluator_evals),
        n_evaluator_evals=charge_evaluator_evals
        + int(density_result.n_evaluator_evals),
        n_cached_nodes=int(
            getattr(density_result, "n_cached_nodes", getattr(density_result, "n_leaf_nodes", 0))
        ),
        n_leaves=int(getattr(density_result, "n_leaves", 0)),
        n_leaf_nodes=int(getattr(density_result, "n_leaf_nodes", 0)),
        subdivisions=charge_refinements + int(getattr(density_result, "subdivisions", 0)),
        charge_integral_atol=charge_integral_atol,
        density_atol=integration.density_matrix_tol,
        density_rtol=0.0,
        error_estimate_available=True,
    )
    return _wrap_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        target_filling=model.filling,
        integration=integration,
    )


def _max_density_change(
    lhs: dict[tuple[int, ...], np.ndarray],
    rhs: dict[tuple[int, ...], np.ndarray],
) -> float:
    keys = frozenset(lhs) | frozenset(rhs)
    if not keys:
        return 0.0
    sample = next(iter(lhs.values()), next(iter(rhs.values())))
    zero = np.zeros_like(sample)
    return max(float(np.max(np.abs(lhs.get(key, zero) - rhs.get(key, zero)))) for key in keys)


def _zero_bdg_matrix(model) -> np.ndarray:
    return np.zeros((2 * model._ndof, 2 * model._ndof), dtype=complex)


def _zero_bdg_array(ndof: int) -> np.ndarray:
    return np.zeros((2 * ndof, 2 * ndof), dtype=complex)


def _zero_electron_matrix(model) -> np.ndarray:
    return np.zeros((model._ndof, model._ndof), dtype=complex)


def _extract_electron_density(density_matrix: _tb_type, model) -> _tb_type:
    return {
        key: matrix[: model._ndof, : model._ndof]
        for key, matrix in density_matrix.items()
    }


def _extract_anomalous_density(density_matrix: _tb_type, model) -> _tb_type:
    return {
        key: matrix[: model._ndof, model._ndof :]
        for key, matrix in density_matrix.items()
    }


def _particle_hole_conjugate(tb: _tb_type) -> _tb_type:
    result = {}
    for key, matrix in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        result[opposite] = -_transpose(matrix)
    return result


def _assemble_bdg_correction(
    normal_block: _tb_type,
    anomalous_block: _tb_type,
    model,
) -> _tb_type:
    zero_e = _zero_electron_matrix(model)
    keys = frozenset(normal_block) | frozenset(anomalous_block)
    hole_block = _particle_hole_conjugate(normal_block)
    assembled = {}
    for key in keys | frozenset(hole_block):
        opposite = tuple(-np.asarray(key, dtype=int))
        normal = normal_block.get(key, zero_e)
        anomalous = anomalous_block.get(key, zero_e)
        lower = _conjugate_transpose(anomalous_block.get(opposite, zero_e))
        hole = hole_block.get(key, zero_e)
        if _is_sparse_like(normal) or _is_sparse_like(anomalous) or _is_sparse_like(hole):
            sparse = _sparse_module()
            assembled[key] = sparse.bmat(
                [
                    [_as_sparse(normal), _as_sparse(anomalous)],
                    [_as_sparse(lower), _as_sparse(hole)],
                ],
                format="csr",
            )
        else:
            assembled[key] = np.block(
                [
                    [np.asarray(normal, dtype=complex), np.asarray(anomalous, dtype=complex)],
                    [np.asarray(lower, dtype=complex), np.asarray(hole, dtype=complex)],
                ]
            )
    return assembled


def _bdg_correction_from_density(density_matrix: _tb_type, model) -> _tb_type:
    electron_density = _extract_electron_density(density_matrix, model)
    anomalous_density = _extract_anomalous_density(density_matrix, model)
    normal_block = normal_meanfield(electron_density, model.h_int)
    zero_e = _zero_electron_matrix(model)
    anomalous_block = {
        key: -_elementwise_product(
            model.h_int.get(key, zero_e),
            anomalous_density.get(key, zero_e),
        )
        for key in frozenset(model.h_int) | frozenset(anomalous_density)
    }
    return _assemble_bdg_correction(normal_block, anomalous_block, model)


def _mix_meanfields(old: _tb_type, new: _tb_type, *, alpha: float, model) -> _tb_type:
    zero = _zero_bdg_matrix(model)
    keys = frozenset(old) | frozenset(new)
    return {
        key: old.get(key, zero) + alpha * (new.get(key, zero) - old.get(key, zero))
        for key in keys
    }


def _flatten_tb(tb: _tb_type) -> np.ndarray:
    if not tb:
        return np.array([], dtype=float)
    ndof = next(iter(tb.values())).shape[0] // 2
    return np.asarray(bdg_tb_to_rparams(tb, ndof), dtype=float)


def _split_bdg_matrix(matrix: Any, ndof: int) -> tuple[Any, Any, Any, Any]:
    array = matrix
    return (
        array[:ndof, :ndof],
        array[:ndof, ndof:],
        array[ndof:, :ndof],
        array[ndof:, ndof:],
    )


def validate_bdg_tb(tb: _tb_type, *, ndof: int, ndim: int, name: str = "BdG correction") -> None:
    expected_shape = (2 * ndof, 2 * ndof)
    zero = _zero_bdg_array(ndof)

    for key, matrix in tb.items():
        if len(key) != ndim:
            raise ValueError(f"{name} keys must match the model dimension")
        if _matrix_shape(matrix) != expected_shape:
            raise ValueError(f"{name} matrices must have shape (2*ndof, 2*ndof)")

    for key, matrix in tb.items():
        opposite = tuple(-np.asarray(key, dtype=int))
        if opposite not in tb:
            raise ValueError(f"{name} must include opposite keys for Hermiticity")
        opposite_matrix = tb[opposite]
        if not _matrix_allclose(matrix, _conjugate_transpose(opposite_matrix)):
            raise ValueError(f"{name} must be Hermitian in real-space tight-binding form")

    keys = frozenset(tb) | {tuple(-np.asarray(key, dtype=int)) for key in tb}
    for key in keys:
        opposite = tuple(-np.asarray(key, dtype=int))
        matrix = tb.get(key, zero)
        opposite_matrix = tb.get(opposite, zero)
        normal, anomalous, lower, hole = _split_bdg_matrix(matrix, ndof)
        opposite_normal, opposite_anomalous, _, _ = _split_bdg_matrix(opposite_matrix, ndof)

        if not _matrix_allclose(hole, -_transpose(opposite_normal)):
            raise ValueError(
                f"{name} lower-right block must equal -h(-R).T in electron-first BdG form"
            )
        if not _matrix_allclose(lower, _conjugate_transpose(opposite_anomalous)):
            raise ValueError(
                f"{name} lower-left block must equal Delta(-R).dagger in electron-first BdG form"
            )


def _bdg_density_keys(model, meanfield: _tb_type) -> list[tuple[int, ...]]:
    del meanfield
    keys = list(model.h_int)
    onsite = (0,) * model._ndim
    if onsite not in keys:
        keys.append(onsite)
    return keys


def _counter_tuple(result: DensityMatrixResult) -> tuple[int, int, int, int, int]:
    info = result.info
    charge_calls = int(getattr(info, "charge_integration_calls", 0) or 0)
    density_calls = int(getattr(info, "density_integration_calls", 0) or 0)
    kernel_evals = int(getattr(info, "n_kernel_evals", 0) or 0)
    unique_evals = int(getattr(info, "unique_evals", kernel_evals) or 0)
    evaluator_evals = int(getattr(info, "n_evaluator_evals", 0) or 0)
    return charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals


def _validate_bdg_correction(model, meanfield: _tb_type) -> None:
    validate_bdg_tb(
        meanfield,
        ndof=model._ndof,
        ndim=model._ndim,
        name="BdG correction",
    )


def solve_bdg_scf(
    model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod,
    scf,
    scf_tol: float,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int | None,
    optimizer,
    optimizer_kwargs: dict | None,
):
    from meanfi.solvers import NoConvergence
    from meanfi._info import SolverResult

    if optimizer is not None or optimizer_kwargs:
        raise NotImplementedError("BdG SCF does not yet support custom optimizers")
    if not isinstance(scf, LinearMixing):
        raise NotImplementedError("BdG SCF currently supports LinearMixing only")
    if scf_tol <= 0:
        raise ValueError("scf_tol must be positive")

    _validate_bdg_correction(model, guess)
    keys = _bdg_density_keys(model, guess)
    meanfield = dict(guess)
    previous_density = None
    last_density_result = None
    residual_norm = float("inf")
    mu_guess = 0.0
    totals = SimpleNamespace(
        charge=0,
        density=0,
        kernel=0,
        unique=0,
        evaluator=0,
    )

    for iteration in range(1, int(scf.max_iterations) + 1):
        density_result = solve_bdg_density_fixed_filling(
            model,
            meanfield,
            keys=keys,
            integration=integration,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            mu_guess=mu_guess,
        )
        charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = _counter_tuple(
            density_result
        )
        totals.charge += charge_calls
        totals.density += density_calls
        totals.kernel += kernel_evals
        totals.unique += unique_evals
        totals.evaluator += evaluator_evals

        if previous_density is not None:
            residual_norm = _max_density_change(
                density_result.density_matrix,
                previous_density,
            )
            if residual_norm <= scf_tol:
                last_density_result = density_result
                info = SCFInfo(
                    method="linear_mixing",
                    iterations=iteration,
                    residual_norm=residual_norm,
                    total_charge_integration_calls=totals.charge,
                    total_density_integration_calls=totals.density,
                    total_kernel_evals=totals.kernel,
                    total_unique_evals=totals.unique,
                    total_evaluator_evals=totals.evaluator,
                )
                return SolverResult(
                    mf=meanfield,
                    density_matrix_result=last_density_result,
                    integration=integration,
                    scf=scf,
                    info=info,
                )

        new_meanfield = _bdg_correction_from_density(
            density_result.density_matrix,
            model,
        )
        _validate_bdg_correction(model, new_meanfield)
        meanfield = _mix_meanfields(
            meanfield,
            new_meanfield,
            alpha=float(scf.alpha),
            model=model,
        )
        _validate_bdg_correction(model, meanfield)
        previous_density = density_result.density_matrix
        last_density_result = density_result
        mu_guess = density_result.mu

    raise NoConvergence(_flatten_tb(meanfield))
