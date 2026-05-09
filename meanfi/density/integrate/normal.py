"""Central density solve pipeline.

The public density APIs should read as one path: construct a physical problem,
build a numerical plan, evaluate at a chemical potential or solve fixed filling,
and return the public result object. Algorithm-specific modules still provide the
low-level matrix-function, quadrature, simplex, and uniform-grid evaluators.
"""

from __future__ import annotations

import numpy as np

from meanfi.density.integrate.common import (
    effective_filling_tol,
    local_density_filling,
    retarget_result_keys,
    uniform_grid_info,
    wrap_adaptive_result,
    wrap_density_result,
)
from meanfi.density.filling import charge_integral_tolerance, mu_bracket, solve_mu
from meanfi.density.kpoint.matrix_functions import (
    DirectDiagonalization,
    RationalFOE,
    resolve_matrix_function,
)
from meanfi.density.kpoint.occupations import fermi_dirac
from meanfi.density.integrate.methods import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    UniformGrid,
)
from meanfi.density.integrate.quadrature.normal import (
    build_normal_backend,
    resolve_normal_matrix_function,
)
from meanfi.density.integrate.quadrature.runtime import (
    build_integrator,
    run_integrator,
    solve_quadrature_at_mu,
)
from meanfi.density.integrate.simplex import (
    density_matrix_at_mu_zero_temp,
    density_matrix_zero_temp,
)
from meanfi.density.integrate.uniform import (
    _tb_point_evaluator,
    build_uniform_grid_node_bundle,
    resolve_uniform_grid_matrix_function,
    solve_uniform_grid_at_mu,
    uniform_grid_density_from_nodes,
    uniform_grid_density_terms,
    uniform_grid_fermi_level,
    uniform_grid_kpoints,
)
from meanfi.density.problem import DensityProblem
from meanfi.density.kpoint.zero_dim import (
    density_from_matrix,
    density_matrix_at_mu_zero_dim,
    zero_dim_zero_temp_mu,
)
from meanfi.density.integrate.workspace import workspace_complex_dtype
from meanfi.results import FixedFillingInfo
from meanfi.space.density_selection import DensitySelection
from meanfi.tb.ops import _tb_type, is_sparse_like, to_dense
from meanfi.tb.transforms import tb_to_kgrid
from meanfi.tb.validate import (
    require_zero_dim_local_key_only,
    tb_dimension,
    tb_orbital_count,
    zero_key,
)


DispatchContext = DensityProblem


def _wrap_adaptive_payload(
    request_context: DensityProblem,
    *,
    density_matrix: _tb_type,
    density_matrix_error: _tb_type | None,
    raw_info: object,
    mu: float,
    filling: float,
    target_filling: float | None,
):
    integration = request_context.integration
    assert isinstance(integration, (AdaptiveQuadrature, AdaptiveSimplex))
    result = wrap_adaptive_result(
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=mu,
        filling=filling,
        target_filling=target_filling,
        integration=integration,
        keys=request_context.solve_keys,
    )
    return retarget_result_keys(result, keys=request_context.requested_keys)


def _normal_zero_dim_fixed_filling(
    matrix: np.ndarray,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    mu_guess: float,
    charge_tol: float,
    mu_xtol: float,
    max_charge_evaluations: int | None,
    density_atol: float,
    density_rtol: float,
    density_selection: DensitySelection | None = None,
    matrix_function: object | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> tuple[_tb_type, _tb_type, float, FixedFillingInfo]:
    resolved_matrix_function = resolve_matrix_function(matrix_function)
    eigenvalues = eigenvectors = None
    if not isinstance(resolved_matrix_function, RationalFOE):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    if kT == 0:
        assert eigenvalues is not None
        assert eigenvectors is not None
        mu, charge = zero_dim_zero_temp_mu(
            eigenvalues,
            filling=filling,
            mu_guess=mu_guess,
        )
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
        rho, error = density_from_matrix(density, keys, density_selection)
        info = FixedFillingInfo(
            mu=mu,
            charge=charge,
            charge_error=abs(charge - filling),
            dcharge_dmu=0.0,
            charge_evaluations=1,
            charge_integration_calls=1,
            density_integration_calls=1,
            charge_n_kernel_evals=1,
            density_n_kernel_evals=0,
            n_kernel_evals=1,
            unique_evals=1,
            charge_n_evaluator_evals=1,
            density_n_evaluator_evals=1,
            n_evaluator_evals=2,
            n_cached_nodes=1,
            n_leaves=1,
            n_leaf_nodes=1,
            subdivisions=0,
            charge_integral_atol=charge_tol,
            density_atol=density_atol,
            density_rtol=density_rtol,
            error_estimate_available=True,
        )
        return rho, error, mu, info

    if isinstance(resolved_matrix_function, RationalFOE):
        raise ValueError(
            "Dense RationalFOE is unsupported; use DirectDiagonalization for dense matrices "
            "or sparse matrices for the MUMPS selected-inverse RationalFOE path"
        )
    assert eigenvalues is not None
    assert eigenvectors is not None
    charge_calls = 0

    def evaluate_charge(candidate_mu: float) -> tuple[float, float, float]:
        nonlocal charge_calls
        charge_calls += 1
        occupation = fermi_dirac(eigenvalues, kT, candidate_mu)
        charge = float(np.sum(occupation))
        derivative = float(np.sum(occupation * (1.0 - occupation) / kT))
        return charge, 0.0, derivative

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=lambda: mu_bracket({tuple(): matrix}, kT),
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=charge_tol,
        mu_tol=mu_xtol,
        max_charge_evaluations=max_charge_evaluations,
    )

    occupation = fermi_dirac(eigenvalues, kT, root.mu)
    density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
    rho, error = density_from_matrix(density, keys, density_selection)

    charge_integral_atol, _ = charge_integral_tolerance(charge_tol)
    info = FixedFillingInfo(
        mu=root.mu,
        charge=root.charge,
        charge_error=root.charge_error,
        dcharge_dmu=float("nan") if root.derivative is None else root.derivative,
        charge_evaluations=root.charge_evaluations,
        charge_integration_calls=charge_calls,
        density_integration_calls=1,
        charge_n_kernel_evals=1,
        density_n_kernel_evals=0,
        n_kernel_evals=1,
        unique_evals=1,
        charge_n_evaluator_evals=charge_calls,
        density_n_evaluator_evals=1,
        n_evaluator_evals=charge_calls + 1,
        n_cached_nodes=1,
        n_leaves=1,
        n_leaf_nodes=1,
        subdivisions=0,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=density_rtol,
        error_estimate_available=True,
    )
    return rho, error, root.mu, info


def _adaptive_quadrature_at_mu(context: DispatchContext, mu: float):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveQuadrature)
    resolve_normal_matrix_function(
        getattr(integration, "matrix_function", None), hamiltonian
    )

    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            hamiltonian[tuple()],
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
            density_selection=None,
            matrix_function=getattr(integration, "matrix_function", None),
            workspace_dtype=workspace_complex_dtype(integration),
            density_tolerance=integration.density_matrix_tol,
        )
    else:
        backend = build_normal_backend(
            hamiltonian,
            integration=integration,
            keys=context.solve_keys,
            kT=context.kT,
            density_selection=context.density_selection,
        )
        density_matrix, density_matrix_error, raw_info = solve_quadrature_at_mu(
            backend,
            mu=mu,
            rule=integration.rule,
            batch_size=integration.batch_size,
            density_atol=integration.density_matrix_tol,
            max_subdivisions=integration.max_refinements,
            error_message="Adaptive quadrature did not converge",
        )

    return _wrap_adaptive_payload(
        context,
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=mu,
        filling=local_density_filling(
            density_matrix,
            local_key=zero_key(tb_dimension(hamiltonian)),
        ),
        target_filling=None,
    )


def _solve_quadrature_fixed_filling(
    backend,
    *,
    filling: float,
    mu_guess: float,
    rule: str,
    batch_size: int | None,
    filling_tol: float,
    mu_tol: float,
    max_charge_evaluations: int | None,
    density_atol: float,
    max_subdivisions: int | None,
    root_error_message: str,
    density_error_message: str,
) -> tuple[_tb_type, _tb_type, FixedFillingInfo]:
    """Run the shared fixed-filling executor for an adaptive backend."""

    charge_integral_atol, charge_integral_rtol = charge_integral_tolerance(filling_tol)
    charge_integrator = build_integrator(
        backend,
        evaluator=backend.charge_evaluator,
        rule=rule,
        batch_size=batch_size,
    )

    charge_integration_calls = 0
    charge_kernel_evals = 0
    charge_evaluator_evals = 0
    charge_subdivisions = 0

    def evaluate_charge(candidate_mu: float) -> tuple[float, float, float | None]:
        nonlocal \
            charge_integration_calls, \
            charge_kernel_evals, \
            charge_evaluator_evals, \
            charge_subdivisions
        result = run_integrator(
            charge_integrator,
            candidate_mu,
            atol=charge_integral_atol,
            rtol=charge_integral_rtol,
            max_subdivisions=max_subdivisions,
            error_message=root_error_message,
        )
        charge_integration_calls += 1
        charge_kernel_evals += int(result.n_kernel_evals)
        charge_evaluator_evals += int(result.n_evaluator_evals)
        charge_subdivisions += int(getattr(result, "subdivisions", 0))
        return backend.split_charge_result(result.estimate, result.error)

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=backend.mu_bracket,
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        use_derivative=backend.charge_has_derivative,
    )

    density_integrator = charge_integrator.replace_evaluator(backend.density_evaluator)
    density_max_subdivisions = 0 if backend.freeze_density_mesh else max_subdivisions
    density_statuses = (
        ("converged", "max_subdivisions")
        if backend.freeze_density_mesh
        else ("converged",)
    )
    density_result = run_integrator(
        density_integrator,
        root.mu,
        atol=density_atol,
        rtol=0.0,
        max_subdivisions=density_max_subdivisions,
        error_message=density_error_message,
        accepted_statuses=density_statuses,
    )
    density_matrix, density_matrix_error = backend.split_density_result(
        density_result.estimate,
        density_result.error,
    )
    raw_info = backend.fixed_filling_info_builder(
        mu=root.mu,
        charge=root.charge,
        charge_error=root.charge_error,
        derivative=float("nan") if root.derivative is None else root.derivative,
        charge_evaluations=root.charge_evaluations,
        charge_integration_calls=charge_integration_calls,
        charge_kernel_evals=charge_kernel_evals,
        charge_evaluator_evals=charge_evaluator_evals,
        charge_subdivisions=charge_subdivisions,
        density_result=density_result,
        charge_integral_atol=charge_integral_atol,
        density_atol=density_atol,
        density_rtol=0.0,
    )
    return density_matrix, density_matrix_error, raw_info


def _adaptive_quadrature_fixed_filling(
    context: DispatchContext,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveQuadrature)

    resolved_filling_tol = effective_filling_tol(
        integration,
        hamiltonian=hamiltonian,
        filling_tol=filling_tol,
    )
    resolve_normal_matrix_function(
        getattr(integration, "matrix_function", None), hamiltonian
    )
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, _mu, raw_info = (
            _normal_zero_dim_fixed_filling(
                matrix=hamiltonian[tuple()],
                filling=filling,
                kT=context.kT,
                keys=context.solve_keys,
                mu_guess=mu_guess,
                charge_tol=resolved_filling_tol,
                mu_xtol=mu_tol,
                max_charge_evaluations=max_charge_evaluations,
                density_atol=integration.density_matrix_tol,
                density_rtol=0.0,
                density_selection=None,
                matrix_function=getattr(integration, "matrix_function", None),
                workspace_dtype=workspace_complex_dtype(integration),
            )
        )
    else:
        backend = build_normal_backend(
            hamiltonian,
            integration=integration,
            keys=context.solve_keys,
            kT=context.kT,
            fixed_filling_tolerance=resolved_filling_tol,
            density_selection=context.density_selection,
        )
        density_matrix, density_matrix_error, raw_info = (
            _solve_quadrature_fixed_filling(
                backend,
                filling=filling,
                mu_guess=mu_guess,
                rule=integration.rule,
                batch_size=integration.batch_size,
                filling_tol=resolved_filling_tol,
                mu_tol=mu_tol,
                max_charge_evaluations=max_charge_evaluations,
                density_atol=integration.density_matrix_tol,
                max_subdivisions=integration.max_refinements,
                root_error_message=(
                    "Adaptive quadrature did not converge while solving for the chemical potential"
                ),
                density_error_message=(
                    "Adaptive quadrature did not converge while evaluating density"
                ),
            )
        )

    return _wrap_adaptive_payload(
        context,
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
    )


def _adaptive_simplex_at_mu(context: DispatchContext, mu: float):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveSimplex)

    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            hamiltonian[tuple()],
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
        )
    else:
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_temp(
            hamiltonian,
            mu=mu,
            keys=context.solve_keys,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            max_subdivisions=integration.max_refinements,
            refinement_depth=integration.refinement_depth,
        )

    return _wrap_adaptive_payload(
        context,
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=mu,
        filling=local_density_filling(
            density_matrix,
            local_key=zero_key(tb_dimension(hamiltonian)),
        ),
        target_filling=None,
    )


def _adaptive_simplex_fixed_filling(
    context: DispatchContext,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
):
    hamiltonian = context.hamiltonian
    integration = context.integration
    assert isinstance(integration, AdaptiveSimplex)

    resolved_filling_tol = effective_filling_tol(
        integration,
        hamiltonian=hamiltonian,
        filling_tol=filling_tol,
    )
    if tb_dimension(hamiltonian) == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, _mu, raw_info = (
            _normal_zero_dim_fixed_filling(
                matrix=hamiltonian[tuple()],
                filling=filling,
                kT=context.kT,
                keys=context.solve_keys,
                mu_guess=mu_guess,
                charge_tol=resolved_filling_tol,
                mu_xtol=mu_tol,
                max_charge_evaluations=max_charge_evaluations,
                density_atol=integration.density_matrix_tol,
                density_rtol=0.0,
            )
        )
    else:
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_temp(
            hamiltonian,
            filling=filling,
            keys=context.solve_keys,
            charge_tol=resolved_filling_tol,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            mu_guess=mu_guess,
            mu_xtol=mu_tol,
            max_charge_evaluations=max_charge_evaluations,
            max_subdivisions=integration.max_refinements,
            refinement_depth=integration.refinement_depth,
        )

    return _wrap_adaptive_payload(
        context,
        density_matrix=density_matrix,
        density_matrix_error=density_matrix_error,
        raw_info=raw_info,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
    )


def _uniform_charge_weights(hamiltonian: _tb_type) -> np.ndarray:
    return np.ones(tb_orbital_count(hamiltonian), dtype=float)


def _implicit_uniform_filling_tol(
    hamiltonian: _tb_type,
    integration: UniformGrid,
    filling_tol: float | None,
) -> float:
    if filling_tol is not None:
        return float(filling_tol)
    return float(0.1 * tb_orbital_count(hamiltonian) * integration.density_matrix_tol)


def _uniform_fixed_filling_from_nodes(
    bundle,
    *,
    hamiltonian: _tb_type,
    integration: UniformGrid,
    filling: float,
    mu_guess: float,
    filling_tol: float,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_bracket_builder,
):
    charge_calls = 0
    n_kpoints = len(bundle.nodes)

    def evaluate_charge(mu: float) -> tuple[float, float, float | None]:
        nonlocal charge_calls
        charge_calls += 1
        total_charge = 0.0
        total_derivative = 0.0
        for node in bundle.nodes:
            node_charge, node_derivative = node.charge_and_derivative(mu)
            total_charge += float(node_charge)
            total_derivative += float(node_derivative)
        if n_kpoints:
            total_charge /= float(n_kpoints)
            total_derivative /= float(n_kpoints)
        return total_charge, 0.0, (total_derivative if bundle.use_derivative else None)

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=mu_bracket_builder,
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        use_derivative=bundle.use_derivative,
    )
    density_matrix, resolved_filling = uniform_grid_density_from_nodes(
        bundle, mu=root.mu
    )
    info = uniform_grid_info(
        integration=integration,
        hamiltonian=hamiltonian,
        n_kernel_evals=(charge_calls + 1) * max(1, n_kpoints),
        n_evaluator_evals=(charge_calls + 1) * max(1, n_kpoints),
        charge_evaluations=root.charge_evaluations,
        charge_integration_calls=charge_calls,
        density_integration_calls=1,
        error_estimate_available=False,
    )
    return density_matrix, root.mu, resolved_filling, info


def _uniform_grid_at_mu(context: DispatchContext, mu: float):
    integration = context.integration
    assert isinstance(integration, UniformGrid)
    return retarget_result_keys(
        solve_uniform_grid_at_mu(
            context.hamiltonian,
            mu=mu,
            kT=context.kT,
            keys=context.solve_keys,
            integration=integration,
            density_selection=context.density_selection,
        ),
        keys=context.requested_keys,
    )


def _uniform_grid_fixed_filling(
    context: DispatchContext,
    filling: float,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
):
    integration = context.integration
    assert isinstance(integration, UniformGrid)
    hamiltonian = context.hamiltonian
    workspace_dtype = workspace_complex_dtype(integration)
    matrix_function = resolve_uniform_grid_matrix_function(
        getattr(integration, "matrix_function", None),
        hamiltonian,
        kT=context.kT,
    )
    if context.kT == 0 and isinstance(matrix_function, DirectDiagonalization):
        if any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
            point_matrix = _tb_point_evaluator(
                hamiltonian, workspace_dtype=workspace_dtype
            )
            eigenvalues = np.stack(
                [
                    np.linalg.eigvalsh(
                        np.asarray(
                            to_dense(point_matrix(kpoint)), dtype=workspace_dtype
                        )
                    )
                    for kpoint in uniform_grid_kpoints(
                        tb_dimension(hamiltonian), integration.nk
                    )
                ],
                axis=0,
            )
        else:
            kham = tb_to_kgrid(hamiltonian, nk=integration.nk)
            eigenvalues = np.linalg.eigvalsh(kham)
        mu = uniform_grid_fermi_level(eigenvalues, filling)
        if any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
            bundle = build_uniform_grid_node_bundle(
                hamiltonian,
                kT=context.kT,
                nk=integration.nk,
                keys=context.solve_keys,
                matrix_function=matrix_function,
                q_diag=_uniform_charge_weights(hamiltonian),
                trace_weights_diag=_uniform_charge_weights(hamiltonian),
                charge_tolerance=integration.density_matrix_tol,
                density_tolerance=integration.density_matrix_tol,
                density_selection=context.density_selection,
                workspace_dtype=workspace_dtype,
            )
            density_matrix, resolved_filling = uniform_grid_density_from_nodes(
                bundle, mu=mu
            )
        else:
            density_matrix, resolved_filling = uniform_grid_density_terms(
                hamiltonian,
                mu=mu,
                kT=0.0,
                nk=integration.nk,
                density_selection=context.density_selection,
                workspace_dtype=workspace_dtype,
            )
        return retarget_result_keys(
            wrap_density_result(
                density_matrix=density_matrix,
                density_matrix_error=None,
                mu=mu,
                filling=resolved_filling,
                target_filling=filling,
                integration=integration,
                info=uniform_grid_info(
                    integration=integration,
                    hamiltonian=hamiltonian,
                    charge_integration_calls=1,
                    density_integration_calls=1,
                    charge_evaluations=1,
                    error_estimate_available=False,
                ),
                keys=context.solve_keys,
            ),
            keys=context.requested_keys,
        )

    resolved_filling_tol = _implicit_uniform_filling_tol(
        hamiltonian,
        integration,
        filling_tol,
    )
    bundle = build_uniform_grid_node_bundle(
        hamiltonian,
        kT=context.kT,
        nk=integration.nk,
        keys=context.solve_keys,
        matrix_function=matrix_function,
        q_diag=_uniform_charge_weights(hamiltonian),
        trace_weights_diag=_uniform_charge_weights(hamiltonian),
        charge_tolerance=resolved_filling_tol,
        density_tolerance=integration.density_matrix_tol,
        density_selection=context.density_selection,
        workspace_dtype=workspace_dtype,
    )
    density_matrix, mu, resolved_filling, info = _uniform_fixed_filling_from_nodes(
        bundle,
        hamiltonian=hamiltonian,
        integration=integration,
        filling=filling,
        mu_guess=mu_guess,
        filling_tol=resolved_filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_bracket_builder=lambda: mu_bracket(hamiltonian, context.kT),
    )
    return retarget_result_keys(
        wrap_density_result(
            density_matrix=density_matrix,
            density_matrix_error=None,
            mu=mu,
            filling=resolved_filling,
            target_filling=filling,
            integration=integration,
            info=info,
            keys=context.solve_keys,
        ),
        keys=context.requested_keys,
    )


# BdG problem construction and evaluator planning.
