from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from meanfi._info import DensityMatrixResult, SCFInfo, SolverResult
from meanfi.integration import IntegrationMethod, solve_density_matrix_fixed_filling
from meanfi.mf import meanfield
from meanfi.model import Model
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.scf import AndersonMixing, LinearMixing, SCFMethod
from meanfi.tb.tb import _tb_type


class NoConvergence(Exception):
    """Raised when the self-consistent field solver does not converge."""

    def __init__(self, last_iterate: "np.ndarray"):
        self.last_iterate = np.array(last_iterate, copy=True)
        super().__init__(self.last_iterate)


@dataclass
class _ScfState:
    iterations: int = 0
    mu: float = 0.0
    density_matrix_result: DensityMatrixResult | None = None
    residual_norm: float = float("inf")
    total_charge_integration_calls: int = 0
    total_density_integration_calls: int = 0
    total_kernel_evals: int = 0
    total_unique_evals: int = 0
    total_evaluator_evals: int = 0


def _max_norm(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _integration_counters(result: DensityMatrixResult) -> tuple[int, int, int, int, int]:
    info = result.info
    return (
        int(getattr(info, "charge_integration_calls", 0) or 0),
        int(getattr(info, "density_integration_calls", 0) or 0),
        int(getattr(info, "n_kernel_evals", 0) or 0),
        int(
            getattr(
                info,
                "unique_evals",
                getattr(info, "n_kernel_evals", getattr(info, "n_kpoints", 0)),
            )
            or 0
        ),
        int(getattr(info, "n_evaluator_evals", 0) or 0),
    )


def _record_density_result(state: _ScfState, result: DensityMatrixResult) -> None:
    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = _integration_counters(
        result
    )
    state.density_matrix_result = result
    state.mu = result.mu
    state.total_charge_integration_calls += charge_calls
    state.total_density_integration_calls += density_calls
    state.total_kernel_evals += kernel_evals
    state.total_unique_evals += unique_evals
    state.total_evaluator_evals += evaluator_evals


def _density_for_hamiltonian(
    model: Model,
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int,
    mu_guess: float,
) -> DensityMatrixResult:
    return solve_density_matrix_fixed_filling(
        hamiltonian,
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=mu_guess,
    )


def _residual(
    density_params: np.ndarray,
    *,
    model: Model,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_mu_iterations: int,
    state: _ScfState,
) -> np.ndarray:
    density_guess = rparams_to_tb(density_params, keys, model._ndof)
    density_result = _density_for_hamiltonian(
        model,
        model.hamiltonian_from_rho(density_guess),
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=state.mu,
    )
    density_new = {key: density_result.density_matrix[key] for key in keys}
    residual = np.asarray(tb_to_rparams(density_new) - density_params, dtype=float).real

    _record_density_result(state, density_result)
    state.residual_norm = _max_norm(residual)
    return residual


def _solve_linear_mixing(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    alpha: float,
    maxiter: int,
    scf_tol: float,
    state: _ScfState,
) -> np.ndarray:
    x = np.array(x0, copy=True)
    for iteration in range(1, maxiter + 1):
        residual = residual_fn(x)
        state.iterations = iteration
        if _max_norm(residual) <= scf_tol:
            return x
        x = x + alpha * residual
    raise NoConvergence(x)


def _translate_no_convergence(exc: Exception, fallback: np.ndarray) -> None:
    if exc.__class__.__name__ != "NoConvergence":
        return

    if hasattr(exc, "last_iterate"):
        last_iterate = exc.last_iterate
    elif exc.args:
        last_iterate = exc.args[0]
    else:
        last_iterate = fallback
    raise NoConvergence(np.asarray(last_iterate, dtype=float)) from exc


def _solve_with_optimizer(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    optimizer: Callable,
    optimizer_kwargs: dict,
    state: _ScfState,
) -> np.ndarray:
    def optimizer_callback(x: np.ndarray, f: np.ndarray) -> None:
        del x
        residual = np.asarray(f, dtype=float)
        state.iterations += 1
        state.residual_norm = _max_norm(residual)

    try:
        with np.errstate(invalid="ignore"):
            result = optimizer(
                residual_fn, x0, callback=optimizer_callback, **optimizer_kwargs
            )
    except TypeError as exc:
        if "callback" not in str(exc):
            raise
        try:
            with np.errstate(invalid="ignore"):
                result = optimizer(residual_fn, x0, **optimizer_kwargs)
        except Exception as inner_exc:  # pragma: no cover - exercised through scipy
            _translate_no_convergence(inner_exc, x0)
            raise
        residual = residual_fn(result)
        state.iterations = max(1, state.iterations)
        state.residual_norm = _max_norm(residual)
        return np.asarray(result, dtype=float)
    except Exception as exc:  # pragma: no cover - exercised through scipy
        _translate_no_convergence(exc, x0)
        raise

    state.iterations = max(1, state.iterations)
    return np.asarray(result, dtype=float)


def _optimizer_name(optimizer: Callable | None) -> str:
    if optimizer is None:
        return "linear_mixing"
    return getattr(optimizer, "__name__", optimizer.__class__.__name__)


def _scf_method_name(scf: SCFMethod) -> str:
    if isinstance(scf, AndersonMixing):
        return "anderson_mixing"
    if isinstance(scf, LinearMixing):
        return "linear_mixing"
    return scf.__class__.__name__


def _optimizer_from_scf(scf: SCFMethod) -> tuple[Callable | None, dict]:
    if isinstance(scf, LinearMixing):
        return None, {"alpha": float(scf.alpha)}
    if isinstance(scf, AndersonMixing):
        try:
            from scipy.optimize import anderson
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise ImportError(
                "AndersonMixing requires scipy to be installed"
            ) from exc
        return anderson, {
            "M": int(scf.M),
            "line_search": scf.line_search,
            "maxiter": int(scf.max_iterations),
        }
    raise TypeError("scf must be an SCFMethod instance")


def _apply_optimizer_defaults(
    optimizer_name: str,
    optimizer_kwargs: dict,
    *,
    scf_tol: float,
    max_iterations: int,
) -> dict:
    kwargs = dict(optimizer_kwargs)
    if optimizer_name == "anderson":
        kwargs.setdefault("M", 0)
        kwargs.setdefault("line_search", "wolfe")
        kwargs.setdefault("maxiter", max_iterations)
        kwargs.setdefault("f_tol", scf_tol)
        kwargs.setdefault("tol_norm", _max_norm)
    return kwargs


def solver(
    model: Model,
    guess: _tb_type,
    *,
    integration: IntegrationMethod,
    scf: SCFMethod = LinearMixing(),
    scf_tol: float = 1e-5,
    filling_tol: float | None = None,
    mu_tol: float = 1e-10,
    max_mu_iterations: int = 128,
    optimizer: Callable | None = None,
    optimizer_kwargs: dict | None = None,
) -> SolverResult:
    """Solve for the self-consistent mean-field correction."""

    if scf_tol <= 0:
        raise ValueError("scf_tol must be positive")
    if getattr(model, "superconducting", False):
        from meanfi._bdg import solve_bdg_scf

        return solve_bdg_scf(
            model,
            guess,
            integration=integration,
            scf=scf,
            scf_tol=scf_tol,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

    keys = list(model.h_int)
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)

    density_matrix_result = _density_for_hamiltonian(
        model,
        model.hamiltonian_from_meanfield(guess),
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=0.0,
    )
    density_guess = {key: density_matrix_result.density_matrix[key] for key in keys}
    density_params0 = tb_to_rparams(density_guess)

    state = _ScfState(mu=density_matrix_result.mu, density_matrix_result=density_matrix_result)
    _record_density_result(state, density_matrix_result)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        return _residual(
            params,
            model=model,
            keys=keys,
            integration=integration,
            filling_tol=filling_tol,
            mu_tol=mu_tol,
            max_mu_iterations=max_mu_iterations,
            state=state,
        )

    if optimizer is None:
        optimizer_impl, default_optimizer_kwargs = _optimizer_from_scf(scf)
        optimizer_name = _scf_method_name(scf)
        if optimizer_impl is None:
            alpha = float(default_optimizer_kwargs.pop("alpha"))
            if optimizer_kwargs:
                invalid_keys = ", ".join(sorted(optimizer_kwargs))
                raise ValueError(
                    f"Unsupported optimizer_kwargs for LinearMixing: {invalid_keys}"
                )
            result_params = _solve_linear_mixing(
                residual_fn,
                density_params0,
                alpha=alpha,
                maxiter=int(scf.max_iterations),
                scf_tol=scf_tol,
                state=state,
            )
        else:
            combined_kwargs = {**default_optimizer_kwargs, **optimizer_kwargs}
            result_params = _solve_with_optimizer(
                residual_fn,
                density_params0,
                optimizer=optimizer_impl,
                optimizer_kwargs=_apply_optimizer_defaults(
                    _optimizer_name(optimizer_impl),
                    combined_kwargs,
                    scf_tol=scf_tol,
                    max_iterations=int(scf.max_iterations),
                ),
                state=state,
            )
    else:
        optimizer_name = _optimizer_name(optimizer)
        result_params = _solve_with_optimizer(
            residual_fn,
            density_params0,
            optimizer=optimizer,
            optimizer_kwargs=_apply_optimizer_defaults(
                optimizer_name,
                optimizer_kwargs,
                scf_tol=scf_tol,
                max_iterations=int(scf.max_iterations),
            ),
            state=state,
        )

    density_result_guess = rparams_to_tb(result_params, keys, model._ndof)
    density_matrix_result = _density_for_hamiltonian(
        model,
        model.hamiltonian_from_rho(density_result_guess),
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_mu_iterations=max_mu_iterations,
        mu_guess=state.mu,
    )
    density_reduced = {
        key: density_matrix_result.density_matrix[key]
        for key in keys
    }
    residual_norm = _max_norm(tb_to_rparams(density_reduced) - result_params)
    mf_result = meanfield(density_reduced, model.h_int)
    tb_result = dict(mf_result)
    tb_result[model._local_key] = tb_result.get(
        model._local_key,
        np.zeros((model._ndof, model._ndof), dtype=complex),
    ) - density_matrix_result.mu * np.eye(model._ndof)

    charge_calls, density_calls, kernel_evals, unique_evals, evaluator_evals = _integration_counters(
        density_matrix_result
    )
    info = SCFInfo(
        method=optimizer_name,
        iterations=max(1, state.iterations),
        residual_norm=residual_norm,
        total_charge_integration_calls=state.total_charge_integration_calls
        + charge_calls,
        total_density_integration_calls=state.total_density_integration_calls
        + density_calls,
        total_kernel_evals=state.total_kernel_evals + kernel_evals,
        total_unique_evals=state.total_unique_evals + unique_evals,
        total_evaluator_evals=state.total_evaluator_evals + evaluator_evals,
    )
    return SolverResult(
        mf=tb_result,
        density_matrix_result=density_matrix_result,
        integration=integration,
        scf=scf,
        info=info,
    )
