from __future__ import annotations

import numpy as np

from meanfi._finite_temp import (
    build_integrator,
    charge_evaluator,
    charge_integral_tolerance,
    density_evaluator,
    fermi_dirac,
    integration_stats,
    mu_bracket,
    run_integrator,
    solve_mu,
    split_charge_result,
    split_density_result,
    expand_mu_bracket,
)
from meanfi._info import DensityIntegrationInfo, FixedFillingInfo
from meanfi._validation import (
    normalize_keys,
    require_zero_dim_local_key_only,
    tb_dimension,
    tb_orbital_count,
    zero_key,
)
from meanfi._zero_dim import density_matrix_at_mu_zero_dim, density_matrix_zero_dim
from meanfi.tb.tb import add_tb, _tb_type


def density_matrix_at_mu(
    h: _tb_type,
    mu: float,
    kT: float,
    keys: list,
    *,
    density_atol: float = 1e-6,
    density_rtol: float = 0.0,
    max_subdivisions: int | None = None,
    rule: str = "auto",
    batch_size: int | None = None,
) -> tuple[_tb_type, _tb_type, DensityIntegrationInfo]:
    """Compute the real-space density matrix at a fixed chemical potential.

    Parameters
    ----------
    h :
        Tight-binding Hamiltonian.
    mu :
        Chemical potential.
    kT :
        Temperature in energy units.
    keys :
        Real-space keys to return in the density matrix.
    density_atol, density_rtol :
        Absolute and relative tolerances for the density integral.
    max_subdivisions :
        Optional backend safety cap. ``None`` leaves the backend unbounded.
    rule, batch_size :
        Advanced finite-temperature quadrature controls.

    Returns
    -------
    :
        Density matrix, error estimate, and integration statistics.
    """

    if kT < 0:
        raise ValueError("density_matrix_at_mu requires kT >= 0")

    normalized_keys = normalize_keys(h, keys)
    ndim = tb_dimension(h)
    if ndim == 0:
        require_zero_dim_local_key_only(h)
        return density_matrix_at_mu_zero_dim(
            h[tuple()], mu=mu, kT=kT, keys=normalized_keys
        )

    if kT == 0:
        if rule != "auto" or batch_size is not None:
            raise ValueError("rule and batch_size are supported only for kT > 0")
        from meanfi.zero_temp import density_matrix_at_mu_zero_temp

        return density_matrix_at_mu_zero_temp(
            h,
            mu=mu,
            keys=normalized_keys,
            density_atol=density_atol,
            density_rtol=density_rtol,
            max_subdivisions=max_subdivisions,
        )

    ndof = tb_orbital_count(h)
    integrator = build_integrator(
        h,
        evaluator=density_evaluator(ndim, ndof, normalized_keys, kT),
        rule=rule,
        batch_size=batch_size,
    )
    result = run_integrator(
        integrator,
        mu,
        atol=density_atol,
        rtol=density_rtol,
        max_subdivisions=max_subdivisions,
        error_message="Adaptive quadrature did not converge",
    )
    rho, error = split_density_result(
        result.estimate, result.error, ndof, normalized_keys
    )
    return rho, error, integration_stats(result)


def density_matrix(
    h: _tb_type,
    filling: float,
    kT: float,
    keys: list,
    *,
    charge_tol: float = 1e-6,
    density_atol: float = 1e-6,
    density_rtol: float = 0.0,
    mu_guess: float = 0.0,
    mu_xtol: float | None = None,
    max_mu_iterations: int = 64,
    max_subdivisions: int | None = None,
    rule: str = "auto",
    batch_size: int | None = None,
) -> tuple[_tb_type, _tb_type, float, FixedFillingInfo]:
    """Compute the fixed-filling real-space density matrix.

    Parameters
    ----------
    h :
        Tight-binding Hamiltonian.
    filling :
        Number of particles per unit cell.
    kT :
        Temperature in energy units.
    keys :
        Real-space keys to return in the density matrix.
    charge_tol, density_atol, density_rtol :
        Accuracy controls for the charge solve and density integral.
    mu_guess :
        Initial guess for the chemical potential.
    mu_xtol :
        Optional tolerance for the chemical-potential root solve. Defaults to
        ``charge_tol`` when omitted.
    max_mu_iterations :
        Maximum number of safeguarded Newton steps.
    max_subdivisions :
        Optional backend safety cap. ``None`` leaves the backend unbounded.
    rule, batch_size :
        Advanced finite-temperature quadrature controls.

    Returns
    -------
    :
        Density matrix, error estimate, solved chemical potential, and solve statistics.
    """

    if kT < 0:
        raise ValueError("density_matrix requires kT >= 0")

    normalized_keys = normalize_keys(h, keys)
    resolved_mu_xtol = float(charge_tol if mu_xtol is None else mu_xtol)
    ndim = tb_dimension(h)
    if ndim == 0:
        require_zero_dim_local_key_only(h)
        return density_matrix_zero_dim(
            h[tuple()],
            filling=filling,
            kT=kT,
            keys=normalized_keys,
            mu_guess=mu_guess,
            charge_tol=charge_tol,
            mu_xtol=resolved_mu_xtol,
            max_mu_iterations=max_mu_iterations,
            density_atol=density_atol,
            density_rtol=density_rtol,
        )

    if kT == 0:
        if rule != "auto" or batch_size is not None:
            raise ValueError("rule and batch_size are supported only for kT > 0")
        from meanfi.zero_temp import density_matrix_zero_temp

        return density_matrix_zero_temp(
            h,
            filling=filling,
            keys=normalized_keys,
            charge_tol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            mu_guess=mu_guess,
            mu_xtol=resolved_mu_xtol,
            max_mu_iterations=max_mu_iterations,
            max_subdivisions=max_subdivisions,
        )

    ndof = tb_orbital_count(h)
    charge_integral_atol, charge_integral_rtol = charge_integral_tolerance(charge_tol)
    charge_integrator = build_integrator(
        h,
        evaluator=charge_evaluator(ndim, ndof, kT),
        rule=rule,
        batch_size=batch_size,
    )

    charge_integration_calls = 0
    charge_kernel_evals = 0
    charge_evaluator_evals = 0

    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        nonlocal charge_integration_calls, charge_kernel_evals, charge_evaluator_evals
        result = run_integrator(
            charge_integrator,
            mu,
            atol=charge_integral_atol,
            rtol=charge_integral_rtol,
            max_subdivisions=max_subdivisions,
            error_message=(
                "Adaptive quadrature did not converge while solving for the chemical potential"
            ),
        )
        charge, charge_error, derivative = split_charge_result(
            result.estimate, result.error
        )
        charge_integration_calls += 1
        charge_kernel_evals += int(result.n_kernel_evals)
        charge_evaluator_evals += int(result.n_evaluator_evals)
        return charge, charge_error, derivative

    lower, upper = mu_bracket(h, kT)
    lower, upper = expand_mu_bracket(
        evaluate_charge,
        filling=filling,
        lower=lower,
        upper=upper,
    )
    mu, charge, charge_error, derivative, iteration = solve_mu(
        evaluate_charge,
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        charge_tol=charge_tol,
        mu_xtol=resolved_mu_xtol,
        max_mu_iterations=max_mu_iterations,
    )

    density_integrator = charge_integrator.replace_evaluator(
        density_evaluator(ndim, ndof, normalized_keys, kT)
    )
    density_result = run_integrator(
        density_integrator,
        mu,
        atol=density_atol,
        rtol=density_rtol,
        max_subdivisions=max_subdivisions,
        error_message="Adaptive quadrature did not converge while evaluating density",
    )

    rho, error = split_density_result(
        density_result.estimate,
        density_result.error,
        ndof,
        normalized_keys,
    )
    density_stats = integration_stats(density_result)
    info = FixedFillingInfo(
        mu=mu,
        charge=charge,
        charge_error=charge_error,
        dcharge_dmu=derivative,
        root_iterations=iteration,
        charge_integration_calls=charge_integration_calls,
        density_integration_calls=1,
        charge_n_kernel_evals=charge_kernel_evals,
        density_n_kernel_evals=int(density_result.n_kernel_evals),
        n_kernel_evals=charge_kernel_evals + int(density_result.n_kernel_evals),
        charge_n_evaluator_evals=charge_evaluator_evals,
        density_n_evaluator_evals=int(density_result.n_evaluator_evals),
        n_evaluator_evals=charge_evaluator_evals
        + int(density_result.n_evaluator_evals),
        n_cached_nodes=density_stats.n_cached_nodes,
        n_leaves=density_stats.n_leaves,
        n_leaf_nodes=density_stats.n_leaf_nodes,
        subdivisions=density_stats.subdivisions,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=density_rtol,
        error_estimate_available=True,
    )
    return rho, error, mu, info


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the mean-field correction from a density matrix."""

    onsite_key = zero_key(tb_dimension(density_matrix))
    direct = {
        onsite_key: np.sum(
            np.asarray(
                [
                    np.diag(
                        np.einsum("pp,pn->n", density_matrix[onsite_key], h_int[vector])
                    )
                    for vector in frozenset(h_int)
                ]
            ),
            axis=0,
        )
    }
    exchange = {
        vector: -1 * h_int.get(vector, 0) * density_matrix[vector]
        for vector in frozenset(h_int)
    }
    return add_tb(direct, exchange)
