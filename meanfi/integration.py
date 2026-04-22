from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi._finite_temp import (
    build_integrator,
    charge_evaluator,
    charge_integral_tolerance,
    density_evaluator,
    expand_mu_bracket,
    fermi_dirac,
    integration_stats,
    mu_bracket,
    run_integrator,
    solve_mu,
    split_charge_result,
    split_density_result,
)
from meanfi._info import (
    AdaptiveQuadratureInfo,
    AdaptiveSimplexInfo,
    DensityIntegrationInfo,
    DensityMatrixResult,
    FixedFillingInfo,
    UniformGridInfo,
)
from meanfi._validation import (
    normalize_keys,
    require_zero_dim_local_key_only,
    tb_dimension,
    tb_orbital_count,
    zero_key,
)
from meanfi._zero_dim import density_matrix_at_mu_zero_dim, density_matrix_zero_dim
from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import kgrid_to_tb, tb_to_kgrid

__all__ = [
    "IntegrationMethod",
    "AdaptiveSimplex",
    "AdaptiveQuadrature",
    "UniformGrid",
    "SimplexGrid",
]


@dataclass(frozen=True)
class IntegrationMethod:
    """Base class for Brillouin-zone integration strategies."""


@dataclass(frozen=True)
class AdaptiveSimplex(IntegrationMethod):
    """Adaptive zero-temperature simplicial integration."""

    density_matrix_tol: float = 1e-6
    max_refinements: int | None = None

    def __post_init__(self) -> None:
        if self.density_matrix_tol <= 0:
            raise ValueError("density_matrix_tol must be positive")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative or None")


@dataclass(frozen=True)
class AdaptiveQuadrature(IntegrationMethod):
    """Adaptive finite-temperature quadrature."""

    density_matrix_tol: float = 1e-6
    max_refinements: int | None = None
    rule: str = "auto"
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.density_matrix_tol <= 0:
            raise ValueError("density_matrix_tol must be positive")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be non-negative or None")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")


@dataclass(frozen=True)
class UniformGrid(IntegrationMethod):
    """Uniform zero-temperature k-grid point sampling."""

    nk: int

    def __post_init__(self) -> None:
        if self.nk <= 0:
            raise ValueError("nk must be positive")


@dataclass(frozen=True)
class SimplexGrid(IntegrationMethod):
    """Reserved placeholder for a future simplex-grid method."""

    nk: int

    def __post_init__(self) -> None:
        if self.nk <= 0:
            raise ValueError("nk must be positive")


def _validate_integration_method(integration: IntegrationMethod, *, kT: float) -> None:
    if kT < 0:
        raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")
    if isinstance(integration, AdaptiveSimplex):
        if kT != 0:
            raise ValueError("AdaptiveSimplex requires kT == 0")
        return
    if isinstance(integration, AdaptiveQuadrature):
        if kT <= 0:
            raise ValueError("AdaptiveQuadrature requires kT > 0")
        return
    if isinstance(integration, UniformGrid):
        if kT != 0:
            raise ValueError("UniformGrid requires kT == 0")
        return
    if isinstance(integration, SimplexGrid):
        raise NotImplementedError("SimplexGrid is reserved for a future implementation")
    raise TypeError("integration must be an IntegrationMethod instance")


def _prepare_keys(
    hamiltonian: _tb_type,
    keys: list[tuple[int, ...]],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], tuple[int, ...]]:
    requested_keys = normalize_keys(hamiltonian, keys)
    ndim = tb_dimension(hamiltonian)
    local_key = zero_key(ndim)
    working_keys = list(requested_keys)
    if local_key not in working_keys:
        working_keys.append(local_key)
    return requested_keys, working_keys, local_key


def _trim_density_matrix(
    density_matrix: _tb_type,
    *,
    keys: list[tuple[int, ...]],
) -> _tb_type:
    sample = next(iter(density_matrix.values()))
    zeros = np.zeros_like(sample)
    return {
        key: np.array(density_matrix.get(key, zeros), copy=True)
        for key in keys
    }


def _trim_density_matrix_error(
    density_matrix_error: _tb_type | None,
    *,
    keys: list[tuple[int, ...]],
) -> _tb_type | None:
    if density_matrix_error is None:
        return None
    sample = next(iter(density_matrix_error.values()))
    zeros = np.zeros_like(sample)
    return {
        key: np.array(density_matrix_error.get(key, zeros), copy=True)
        for key in keys
    }


def _local_density_filling(
    density_matrix: _tb_type,
    *,
    local_key: tuple[int, ...],
) -> float:
    return float(np.trace(density_matrix[local_key]).real)


def _effective_filling_tol(
    integration: IntegrationMethod,
    *,
    hamiltonian: _tb_type,
    filling_tol: float | None,
) -> float:
    if filling_tol is not None:
        if filling_tol <= 0:
            raise ValueError("filling_tol must be positive when provided")
        return float(filling_tol)

    if isinstance(integration, (AdaptiveSimplex, AdaptiveQuadrature)):
        return float(tb_orbital_count(hamiltonian) * integration.density_matrix_tol)

    raise ValueError("UniformGrid requires an implicit grid-resolved filling target")


def _translate_adaptive_info(
    integration: AdaptiveSimplex | AdaptiveQuadrature,
    raw_info: DensityIntegrationInfo | FixedFillingInfo,
):
    info_type = (
        AdaptiveSimplexInfo if isinstance(integration, AdaptiveSimplex) else AdaptiveQuadratureInfo
    )
    return info_type(
        n_kernel_evals=int(raw_info.n_kernel_evals),
        n_evaluator_evals=int(raw_info.n_evaluator_evals),
        n_cached_nodes=int(raw_info.n_cached_nodes),
        n_leaves=int(raw_info.n_leaves),
        n_leaf_nodes=int(raw_info.n_leaf_nodes),
        refinements=int(raw_info.subdivisions),
        error_estimate_available=bool(raw_info.error_estimate_available),
        root_iterations=getattr(raw_info, "root_iterations", None),
        charge_integration_calls=getattr(raw_info, "charge_integration_calls", None),
        density_integration_calls=getattr(raw_info, "density_integration_calls", None),
    )


def _uniform_grid_info(
    *,
    integration: UniformGrid,
    hamiltonian: _tb_type,
) -> UniformGridInfo:
    ndim = tb_dimension(hamiltonian)
    n_kpoints = 1 if ndim == 0 else int(integration.nk**ndim)
    return UniformGridInfo(nk=int(integration.nk), n_kpoints=n_kpoints)


def _wrap_density_result(
    *,
    density_matrix: _tb_type,
    density_matrix_error: _tb_type | None,
    mu: float,
    filling: float,
    target_filling: float | None,
    integration: IntegrationMethod,
    info,
    keys: list[tuple[int, ...]],
) -> DensityMatrixResult:
    trimmed_density_matrix = _trim_density_matrix(density_matrix, keys=keys)
    trimmed_density_matrix_error = _trim_density_matrix_error(
        density_matrix_error,
        keys=keys,
    )
    filling_residual = (
        None if target_filling is None else abs(float(filling) - float(target_filling))
    )
    return DensityMatrixResult(
        density_matrix=trimmed_density_matrix,
        density_matrix_error=trimmed_density_matrix_error,
        mu=float(mu),
        filling=float(filling),
        target_filling=None if target_filling is None else float(target_filling),
        filling_residual=filling_residual,
        integration=integration,
        info=info,
    )


def _retarget_result_keys(
    result: DensityMatrixResult,
    *,
    keys: list[tuple[int, ...]],
) -> DensityMatrixResult:
    if list(result.density_matrix) == list(keys):
        return result
    return DensityMatrixResult(
        density_matrix=_trim_density_matrix(result.density_matrix, keys=keys),
        density_matrix_error=_trim_density_matrix_error(
            result.density_matrix_error,
            keys=keys,
        ),
        mu=result.mu,
        filling=result.filling,
        target_filling=result.target_filling,
        filling_residual=result.filling_residual,
        integration=result.integration,
        info=result.info,
    )


def _solve_adaptive_quadrature_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: AdaptiveQuadrature,
) -> DensityMatrixResult:
    ndim = tb_dimension(hamiltonian)
    ndof = tb_orbital_count(hamiltonian)
    if ndim == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            hamiltonian[tuple()],
            mu=mu,
            kT=kT,
            keys=keys,
        )
    else:
        integrator = build_integrator(
            hamiltonian,
            evaluator=density_evaluator(ndim, ndof, keys, kT),
            rule=integration.rule,
            batch_size=integration.batch_size,
        )
        result = run_integrator(
            integrator,
            mu,
            atol=integration.density_matrix_tol,
            rtol=0.0,
            max_subdivisions=integration.max_refinements,
            error_message="Adaptive quadrature did not converge",
        )
        density_matrix, density_matrix_error = split_density_result(
            result.estimate,
            result.error,
            ndof,
            keys,
        )
        raw_info = integration_stats(result)

    public_info = _translate_adaptive_info(integration, raw_info)
    error = (
        density_matrix_error if public_info.error_estimate_available else None
    )
    local_key = zero_key(tb_dimension(hamiltonian))
    filling = _local_density_filling(density_matrix, local_key=local_key)
    return _wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=error,
        mu=mu,
        filling=filling,
        target_filling=None,
        integration=integration,
        info=public_info,
        keys=keys,
    )


def _solve_adaptive_quadrature_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: AdaptiveQuadrature,
    filling_tol: float,
    mu_tol: float,
    max_mu_iterations: int,
    mu_guess: float,
) -> DensityMatrixResult:
    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, mu, raw_info = density_matrix_zero_dim(
            hamiltonian[tuple()],
            filling=filling,
            kT=kT,
            keys=keys,
            mu_guess=mu_guess,
            charge_tol=filling_tol,
            mu_xtol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
        )
    else:
        ndof = tb_orbital_count(hamiltonian)
        charge_integral_atol, charge_integral_rtol = charge_integral_tolerance(
            filling_tol
        )
        charge_integrator = build_integrator(
            hamiltonian,
            evaluator=charge_evaluator(ndim, ndof, kT),
            rule=integration.rule,
            batch_size=integration.batch_size,
        )

        charge_integration_calls = 0
        charge_kernel_evals = 0
        charge_evaluator_evals = 0

        def evaluate_charge(candidate_mu: float) -> tuple[float, float, float]:
            nonlocal charge_integration_calls, charge_kernel_evals, charge_evaluator_evals
            result = run_integrator(
                charge_integrator,
                candidate_mu,
                atol=charge_integral_atol,
                rtol=charge_integral_rtol,
                max_subdivisions=integration.max_refinements,
                error_message=(
                    "Adaptive quadrature did not converge while solving for the chemical potential"
                ),
            )
            charge, charge_error, derivative = split_charge_result(
                result.estimate,
                result.error,
            )
            charge_integration_calls += 1
            charge_kernel_evals += int(result.n_kernel_evals)
            charge_evaluator_evals += int(result.n_evaluator_evals)
            return charge, charge_error, derivative

        lower, upper = mu_bracket(hamiltonian, kT)
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

        density_integrator = charge_integrator.replace_evaluator(
            density_evaluator(ndim, ndof, keys, kT)
        )
        density_result = run_integrator(
            density_integrator,
            mu,
            atol=integration.density_matrix_tol,
            rtol=0.0,
            max_subdivisions=integration.max_refinements,
            error_message="Adaptive quadrature did not converge while evaluating density",
        )
        density_matrix, density_matrix_error = split_density_result(
            density_result.estimate,
            density_result.error,
            ndof,
            keys,
        )
        density_stats = integration_stats(density_result)
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
            charge_n_evaluator_evals=charge_evaluator_evals,
            density_n_evaluator_evals=int(density_result.n_evaluator_evals),
            n_evaluator_evals=charge_evaluator_evals
            + int(density_result.n_evaluator_evals),
            n_cached_nodes=density_stats.n_cached_nodes,
            n_leaves=density_stats.n_leaves,
            n_leaf_nodes=density_stats.n_leaf_nodes,
            subdivisions=density_stats.subdivisions,
            charge_integral_atol=charge_integral_atol,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            error_estimate_available=True,
        )

    public_info = _translate_adaptive_info(integration, raw_info)
    error = density_matrix_error if public_info.error_estimate_available else None
    return _wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=error,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
        integration=integration,
        info=public_info,
        keys=keys,
    )


def _solve_adaptive_simplex_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: AdaptiveSimplex,
) -> DensityMatrixResult:
    from meanfi.zero_temp import density_matrix_at_mu_zero_temp

    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_dim(
            hamiltonian[tuple()],
            mu=mu,
            kT=kT,
            keys=keys,
        )
    else:
        density_matrix, density_matrix_error, raw_info = density_matrix_at_mu_zero_temp(
            hamiltonian,
            mu=mu,
            keys=keys,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            max_subdivisions=integration.max_refinements,
        )

    public_info = _translate_adaptive_info(integration, raw_info)
    error = density_matrix_error if public_info.error_estimate_available else None
    local_key = zero_key(tb_dimension(hamiltonian))
    filling = _local_density_filling(density_matrix, local_key=local_key)
    return _wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=error,
        mu=mu,
        filling=filling,
        target_filling=None,
        integration=integration,
        info=public_info,
        keys=keys,
    )


def _solve_adaptive_simplex_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: AdaptiveSimplex,
    filling_tol: float,
    mu_tol: float,
    max_mu_iterations: int,
    mu_guess: float,
) -> DensityMatrixResult:
    from meanfi.zero_temp import density_matrix_zero_temp

    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        require_zero_dim_local_key_only(hamiltonian)
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_dim(
            hamiltonian[tuple()],
            filling=filling,
            kT=kT,
            keys=keys,
            mu_guess=mu_guess,
            charge_tol=filling_tol,
            mu_xtol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
        )
    else:
        density_matrix, density_matrix_error, _mu, raw_info = density_matrix_zero_temp(
            hamiltonian,
            filling=filling,
            keys=keys,
            charge_tol=filling_tol,
            density_atol=integration.density_matrix_tol,
            density_rtol=0.0,
            mu_guess=mu_guess,
            mu_xtol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            max_subdivisions=integration.max_refinements,
        )

    public_info = _translate_adaptive_info(integration, raw_info)
    error = density_matrix_error if public_info.error_estimate_available else None
    return _wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=error,
        mu=raw_info.mu,
        filling=raw_info.charge,
        target_filling=filling,
        integration=integration,
        info=public_info,
        keys=keys,
    )


def _uniform_grid_density_terms(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    nk: int,
) -> tuple[_tb_type, float]:
    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian[tuple()])
        occupation = fermi_dirac(eigenvalues, kT, mu)
        density = eigenvectors * occupation[np.newaxis, :] @ eigenvectors.conj().T
        return {tuple(): density}, float(np.sum(occupation))

    kham = tb_to_kgrid(hamiltonian, nk=nk)
    eigenvalues, eigenvectors = np.linalg.eigh(kham)
    occupation = fermi_dirac(eigenvalues, kT, mu)
    occupied_vectors = eigenvectors * occupation[..., np.newaxis, :]
    density_matrix_k = occupied_vectors @ eigenvectors.conj().swapaxes(-1, -2)
    density_matrix = kgrid_to_tb(density_matrix_k)
    filling = float(np.mean(np.sum(occupation, axis=-1)))
    return density_matrix, filling


def _uniform_grid_fermi_level(eigenvalues: np.ndarray, filling: float) -> float:
    norbs = eigenvalues.shape[-1]
    flat = np.sort(eigenvalues.reshape(-1))
    n_total = flat.size
    index = int(round(n_total * filling / norbs))
    if index >= n_total:
        return float(flat[-1])
    if index == 0:
        return float(flat[0])
    return float((flat[index - 1] + flat[index]) / 2.0)


def _solve_uniform_grid_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: UniformGrid,
) -> DensityMatrixResult:
    density_matrix, filling = _uniform_grid_density_terms(
        hamiltonian,
        mu=mu,
        kT=kT,
        nk=integration.nk,
    )
    return _wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=None,
        mu=mu,
        filling=filling,
        target_filling=None,
        integration=integration,
        info=_uniform_grid_info(integration=integration, hamiltonian=hamiltonian),
        keys=keys,
    )


def _solve_uniform_grid_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: UniformGrid,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int,
) -> DensityMatrixResult:
    del kT
    if filling_tol is not None or mu_tol != 1e-10 or max_mu_iterations != 128:
        raise ValueError(
            "UniformGrid does not support filling_tol, mu_tol, or max_mu_iterations"
        )

    ndim = tb_dimension(hamiltonian)
    if ndim == 0:
        eigenvalues = np.linalg.eigvalsh(hamiltonian[tuple()])
    else:
        kham = tb_to_kgrid(hamiltonian, nk=integration.nk)
        eigenvalues = np.linalg.eigvalsh(kham)
    mu = _uniform_grid_fermi_level(eigenvalues, filling)
    density_matrix, resolved_filling = _uniform_grid_density_terms(
        hamiltonian,
        mu=mu,
        kT=0.0,
        nk=integration.nk,
    )
    return _wrap_density_result(
        density_matrix=density_matrix,
        density_matrix_error=None,
        mu=mu,
        filling=resolved_filling,
        target_filling=filling,
        integration=integration,
        info=_uniform_grid_info(integration=integration, hamiltonian=hamiltonian),
        keys=keys,
    )


def solve_density_matrix_at_mu(
    hamiltonian: _tb_type,
    *,
    mu: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
) -> DensityMatrixResult:
    _validate_integration_method(integration, kT=kT)
    requested_keys, working_keys, _local_key = _prepare_keys(hamiltonian, keys)

    if isinstance(integration, AdaptiveQuadrature):
        return _retarget_result_keys(
            _solve_adaptive_quadrature_at_mu(
                hamiltonian,
                mu=mu,
                kT=kT,
                keys=requested_keys if working_keys == requested_keys else working_keys,
                integration=integration,
            ),
            keys=requested_keys,
        )
    if isinstance(integration, AdaptiveSimplex):
        return _retarget_result_keys(
            _solve_adaptive_simplex_at_mu(
                hamiltonian,
                mu=mu,
                kT=kT,
                keys=requested_keys if working_keys == requested_keys else working_keys,
                integration=integration,
            ),
            keys=requested_keys,
        )
    if isinstance(integration, UniformGrid):
        return _retarget_result_keys(
            _solve_uniform_grid_at_mu(
                hamiltonian,
                mu=mu,
                kT=kT,
                keys=requested_keys if working_keys == requested_keys else working_keys,
                integration=integration,
            ),
            keys=requested_keys,
        )
    raise NotImplementedError("SimplexGrid is reserved for a future implementation")


def solve_density_matrix_fixed_filling(
    hamiltonian: _tb_type,
    *,
    filling: float,
    kT: float,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int,
    mu_guess: float = 0.0,
) -> DensityMatrixResult:
    _validate_integration_method(integration, kT=kT)
    if mu_tol <= 0:
        raise ValueError("mu_tol must be positive")
    if max_mu_iterations <= 0:
        raise ValueError("max_mu_iterations must be positive")

    requested_keys, working_keys, _local_key = _prepare_keys(hamiltonian, keys)
    solve_keys = requested_keys if working_keys == requested_keys else working_keys

    if isinstance(integration, AdaptiveQuadrature):
        return _retarget_result_keys(
            _solve_adaptive_quadrature_fixed_filling(
                hamiltonian,
                filling=filling,
                kT=kT,
                keys=solve_keys,
                integration=integration,
                filling_tol=_effective_filling_tol(
                    integration,
                    hamiltonian=hamiltonian,
                    filling_tol=filling_tol,
                ),
                mu_tol=mu_tol,
                max_mu_iterations=max_mu_iterations,
                mu_guess=mu_guess,
            ),
            keys=requested_keys,
        )
    if isinstance(integration, AdaptiveSimplex):
        return _retarget_result_keys(
            _solve_adaptive_simplex_fixed_filling(
                hamiltonian,
                filling=filling,
                kT=kT,
                keys=solve_keys,
                integration=integration,
                filling_tol=_effective_filling_tol(
                    integration,
                    hamiltonian=hamiltonian,
                    filling_tol=filling_tol,
                ),
                mu_tol=mu_tol,
                max_mu_iterations=max_mu_iterations,
                mu_guess=mu_guess,
            ),
            keys=requested_keys,
        )
    if isinstance(integration, UniformGrid):
        return _retarget_result_keys(
            _solve_uniform_grid_fixed_filling(
                hamiltonian,
                filling=filling,
                kT=kT,
                keys=solve_keys,
                integration=integration,
                filling_tol=filling_tol,
                mu_tol=mu_tol,
                max_mu_iterations=max_mu_iterations,
            ),
            keys=requested_keys,
        )
    raise NotImplementedError("SimplexGrid is reserved for a future implementation")
