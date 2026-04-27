from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from meanfi.core.filling import mu_bracket
from meanfi.integrate.fixed_filling import solve_fixed_filling_root
from meanfi.tb.tb import _tb_type

from .backend import AdaptiveIntegrator, _GEOM_TOL, build_extension_runtime
from .solve import _density_integration_info, _fixed_filling_info, _nan_density_error_like, _vector_to_density


def _coarse_density_summary(
    *,
    geometry,
    vertex_cache,
    keys: list[tuple[int, ...]],
    mu: float,
):
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        tol=float(_GEOM_TOL),
    )
    estimate, _owner_ids, _owner_estimates, evaluator_evals = integrator.evaluate_density(
        float(mu),
        0,
    )
    estimate = np.asarray(estimate, dtype=complex)
    zero_error = np.zeros(estimate.shape, dtype=float)
    rho, _ = _vector_to_density(estimate, zero_error, int(vertex_cache.ndof), keys)
    return rho, _nan_density_error_like(rho), int(evaluator_evals)


def root_mesh_density_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
):
    geometry, vertex_cache = build_extension_runtime(h)
    rho, error, evaluator_evals = _coarse_density_summary(
        geometry=geometry,
        vertex_cache=vertex_cache,
        keys=keys,
        mu=mu,
    )
    result = SimpleNamespace(
        evaluator_evals=evaluator_evals,
        subdivisions=0,
        n_leaves=int(geometry.n_active),
        n_leaf_nodes=int(geometry.n_leaf_vertices),
        error_estimate_available=False,
    )
    info = _density_integration_info(result=result, spectral_cache=vertex_cache)
    return rho, error, info


def root_mesh_fixed_filling_zero_temp(
    h: _tb_type,
    *,
    filling: float,
    keys: list[tuple[int, ...]],
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
    mu_guess: float,
    mu_xtol: float,
    max_mu_iterations: int | None,
):
    geometry, vertex_cache = build_extension_runtime(h)
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.ascontiguousarray(np.asarray(keys, dtype=np.float64)),
        tol=float(_GEOM_TOL),
    )
    charge_integration_calls = 0
    charge_evaluator_evals = 0

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        nonlocal charge_integration_calls, charge_evaluator_evals
        (
            charge,
            derivative,
            derivative_exact,
            _owner_ids,
            _owner_charges,
            evaluator_evals,
        ) = integrator.evaluate_charge(float(mu), 0)
        charge_integration_calls += 1
        charge_evaluator_evals += int(evaluator_evals)
        resolved_derivative = (
            float(derivative) if bool(derivative_exact) and np.isfinite(derivative) else float("nan")
        )
        return float(charge), 0.0, resolved_derivative

    root = solve_fixed_filling_root(
        evaluate_charge=evaluate_charge,
        mu_bracket=lambda: mu_bracket(h, 0.0),
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=charge_tol,
        mu_tol=mu_xtol,
        max_mu_iterations=max_mu_iterations,
    )

    charge_kernel_evals = int(vertex_cache.n_kernel_evals)
    rho, error, density_evaluator_evals = _coarse_density_summary(
        geometry=geometry,
        vertex_cache=vertex_cache,
        keys=keys,
        mu=root.mu,
    )
    density_kernel_evals = int(vertex_cache.n_kernel_evals) - charge_kernel_evals

    charge_result = SimpleNamespace(
        mu=root.mu,
        charge=root.charge,
        charge_error=float("nan"),
        dcharge_dmu=root.derivative,
        root_iterations=root.root_iterations,
        charge_integration_calls=int(charge_integration_calls),
        evaluator_evals=int(charge_evaluator_evals),
        subdivisions=0,
        error_estimate_available=False,
    )
    density_result = SimpleNamespace(
        evaluator_evals=int(density_evaluator_evals),
        subdivisions=0,
        n_leaves=int(geometry.n_active),
        n_leaf_nodes=int(geometry.n_leaf_vertices),
        error_estimate_available=False,
    )
    info = _fixed_filling_info(
        charge_result=charge_result,
        density_result=density_result,
        spectral_cache=vertex_cache,
        charge_kernel_evals=charge_kernel_evals,
        density_kernel_evals=density_kernel_evals,
        charge_tol=charge_tol,
        density_atol=density_atol,
        density_rtol=density_rtol,
    )
    return rho, error, root.mu, info
