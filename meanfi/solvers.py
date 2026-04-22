from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from meanfi._info import FixedFillingInfo
from meanfi.mf import density_matrix as solve_density_matrix
from meanfi.mf import meanfield
from meanfi.model import Model
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.tb.tb import _tb_type


class NoConvergence(Exception):
    """Raised when the self-consistent field solver does not converge."""

    def __init__(self, last_iterate: "np.ndarray"):
        self.last_iterate = np.array(last_iterate, copy=True)
        super().__init__(self.last_iterate)


@dataclass(frozen=True)
class SolverInfo:
    """Summary of a self-consistent mean-field solve."""

    optimizer: str
    iterations: int
    residual_norm: float
    mu: float
    total_charge_integration_calls: int
    total_density_integration_calls: int
    total_kernel_evals: int
    total_evaluator_evals: int
    last_density_info: FixedFillingInfo


@dataclass
class _ScfState:
    iterations: int = 0
    mu: float = 0.0
    rho: _tb_type | None = None
    info: FixedFillingInfo | None = None
    residual_norm: float = float("inf")
    total_charge_integration_calls: int = 0
    total_density_integration_calls: int = 0
    total_kernel_evals: int = 0
    total_evaluator_evals: int = 0

    def as_dict(self) -> dict:
        """Expose callback state in the legacy dictionary shape."""

        return {
            "iterations": self.iterations,
            "mu": self.mu,
            "rho": self.rho,
            "info": self.info,
            "residual_norm": self.residual_norm,
            "total_charge_integration_calls": self.total_charge_integration_calls,
            "total_density_integration_calls": self.total_density_integration_calls,
            "total_kernel_evals": self.total_kernel_evals,
            "total_evaluator_evals": self.total_evaluator_evals,
        }


def _max_norm(values: np.ndarray) -> float:
    """Return the componentwise maximum absolute value."""

    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _record_iteration(
    iteration: int, residual: np.ndarray, *, callback, state: _ScfState
) -> None:
    state.iterations = iteration
    if callback is not None:
        callback(iteration, residual, state.as_dict())


def _density_for_hamiltonian(
    model: Model,
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    mu_guess: float,
):
    """Run a fixed-filling density solve using the model-level accuracy policy."""

    return solve_density_matrix(
        hamiltonian,
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        charge_tol=model.charge_tol,
        density_atol=model.density_atol,
        density_rtol=model.density_rtol,
        mu_guess=mu_guess,
        mu_xtol=model.mu_xtol,
        max_subdivisions=model.max_subdivisions,
    )


def _residual(
    rho_params: np.ndarray,
    *,
    model: Model,
    keys: list[tuple[int, ...]],
    state: _ScfState,
    debug: bool,
) -> np.ndarray:
    rho_reduced = rparams_to_tb(rho_params, keys, model._ndof)
    rho_new, _, mu, info = model.density_matrix(
        rho_reduced,
        keys=keys,
        mu_guess=state.mu,
    )
    rho_new_reduced = {key: rho_new[key] for key in keys}
    residual = np.asarray(tb_to_rparams(rho_new_reduced) - rho_params, dtype=float).real

    state.mu = mu
    state.rho = rho_new_reduced
    state.info = info
    state.residual_norm = _max_norm(residual)
    state.total_charge_integration_calls += info.charge_integration_calls
    state.total_density_integration_calls += info.density_integration_calls
    state.total_kernel_evals += info.n_kernel_evals
    state.total_evaluator_evals += info.n_evaluator_evals

    if debug:
        message = (
            f"mu={mu:.6g}, dN/dmu={info.dcharge_dmu:.6g}, "
            f"residual={state.residual_norm:.6g}"
        )
        sys.stdout.write("\r" + message)
        sys.stdout.flush()

    return residual


def _solve_linear_mixing(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    alpha: float,
    maxiter: int,
    scf_tol: float,
    callback,
    state: _ScfState,
) -> np.ndarray:
    x = np.array(x0, copy=True)
    for iteration in range(1, maxiter + 1):
        residual = residual_fn(x)
        _record_iteration(iteration, residual, callback=callback, state=state)
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
    callback,
    state: _ScfState,
) -> np.ndarray:
    def optimizer_callback(x: np.ndarray, f: np.ndarray) -> None:
        del x
        residual = np.asarray(f, dtype=float)
        _record_iteration(
            state.iterations + 1, residual, callback=callback, state=state
        )

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
        _record_iteration(
            max(1, state.iterations), residual, callback=callback, state=state
        )
        return np.asarray(result, dtype=float)
    except Exception as exc:  # pragma: no cover - exercised through scipy
        _translate_no_convergence(exc, x0)
        raise

    return np.asarray(result, dtype=float)


def _optimizer_name(optimizer: Callable | None) -> str:
    if optimizer is None:
        return "linear_mixing"
    return getattr(optimizer, "__name__", optimizer.__class__.__name__)


def _apply_optimizer_defaults(
    optimizer_name: str,
    optimizer_kwargs: dict,
    *,
    scf_tol: float,
    max_scf_steps: int,
) -> dict:
    """Fill default convergence knobs for known external optimizers."""

    kwargs = dict(optimizer_kwargs)
    if optimizer_name == "anderson":
        kwargs.setdefault("M", 0)
        kwargs.setdefault("line_search", "wolfe")
        kwargs.setdefault("maxiter", max_scf_steps)
        kwargs.setdefault("f_tol", scf_tol)
        kwargs.setdefault("tol_norm", _max_norm)
    return kwargs


def solver(
    model: Model,
    mf_guess: _tb_type,
    *,
    mu_guess: float = 0.0,
    optimizer: Callable | None = None,
    optimizer_kwargs: dict | None = None,
    max_scf_steps: int = 100,
    callback=None,
    debug: bool = False,
    return_info: bool = False,
) -> _tb_type | tuple[_tb_type, SolverInfo]:
    """Solve for the self-consistent mean-field correction."""

    keys = list(model.h_int)
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)

    rho_guess, _, mu, info = _density_for_hamiltonian(
        model,
        model.hamiltonian_from_meanfield(mf_guess),
        keys=keys,
        mu_guess=mu_guess,
    )
    rho_guess_reduced = {key: rho_guess[key] for key in keys}
    rho_params0 = tb_to_rparams(rho_guess_reduced)

    state = _ScfState(mu=mu, rho=rho_guess_reduced, info=info)
    state.total_charge_integration_calls += info.charge_integration_calls
    state.total_density_integration_calls += info.density_integration_calls
    state.total_kernel_evals += info.n_kernel_evals
    state.total_evaluator_evals += info.n_evaluator_evals

    def residual_fn(params: np.ndarray) -> np.ndarray:
        return _residual(
            params,
            model=model,
            keys=keys,
            state=state,
            debug=debug,
        )

    optimizer_name = _optimizer_name(optimizer)
    if optimizer is None:
        alpha = float(optimizer_kwargs.pop("alpha", 0.5))
        if optimizer_kwargs:
            invalid_keys = ", ".join(sorted(optimizer_kwargs))
            raise ValueError(
                f"Unsupported optimizer_kwargs for internal linear mixing: {invalid_keys}"
            )
        result_params = _solve_linear_mixing(
            residual_fn,
            rho_params0,
            alpha=alpha,
            maxiter=max_scf_steps,
            scf_tol=model.scf_tol,
            callback=callback,
            state=state,
        )
    else:
        result_params = _solve_with_optimizer(
            residual_fn,
            rho_params0,
            optimizer=optimizer,
            optimizer_kwargs=_apply_optimizer_defaults(
                optimizer_name,
                optimizer_kwargs,
                scf_tol=model.scf_tol,
                max_scf_steps=max_scf_steps,
            ),
            callback=callback,
            state=state,
        )

    rho_result = rparams_to_tb(result_params, keys, model._ndof)
    rho_final, _, mu_final, info_final = model.density_matrix(
        rho_result,
        keys=keys,
        mu_guess=state.mu,
    )
    rho_reduced_final = {key: rho_final[key] for key in keys}
    residual_norm = _max_norm(tb_to_rparams(rho_reduced_final) - result_params)
    mf_result = meanfield(rho_reduced_final, model.h_int)
    tb_result = dict(mf_result)
    tb_result[model._local_key] = tb_result.get(
        model._local_key, np.zeros((model._ndof, model._ndof), dtype=complex)
    ) - mu_final * np.eye(model._ndof)

    solver_info = SolverInfo(
        optimizer=optimizer_name,
        iterations=max(1, state.iterations),
        residual_norm=residual_norm,
        mu=mu_final,
        total_charge_integration_calls=state.total_charge_integration_calls
        + info_final.charge_integration_calls,
        total_density_integration_calls=state.total_density_integration_calls
        + info_final.density_integration_calls,
        total_kernel_evals=state.total_kernel_evals + info_final.n_kernel_evals,
        total_evaluator_evals=state.total_evaluator_evals
        + info_final.n_evaluator_evals,
        last_density_info=info_final,
    )

    if debug:
        sys.stdout.write("\n")
    if return_info:
        return tb_result, solver_info
    return tb_result
