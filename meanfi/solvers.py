import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy
from scipy.optimize._nonlin import NoConvergence

from meanfi.mf import FixedFillingInfo, meanfield, density_matrix
from meanfi.model import Model
from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.tb.tb import _tb_type


@dataclass(frozen=True)
class SolverInfo:
    """Summary of a self-consistent mean-field solve."""

    mixing: str
    iterations: int
    residual_norm: float
    mu: float
    total_charge_integration_calls: int
    total_density_integration_calls: int
    total_kernel_evals: int
    total_evaluator_evals: int
    last_density_info: FixedFillingInfo


def _scf_state() -> dict:
    return {
        "iterations": 0,
        "mu": 0.0,
        "rho": None,
        "info": None,
        "residual_norm": np.inf,
        "total_charge_integration_calls": 0,
        "total_density_integration_calls": 0,
        "total_kernel_evals": 0,
        "total_evaluator_evals": 0,
    }


def _residual(
    rho_params: np.ndarray,
    *,
    model: Model,
    keys: list[tuple[int, ...]],
    state: dict,
    debug: bool,
    rule: str,
    batch_size: int | None,
    max_mu_iterations: int,
    max_subdivisions: int | None,
) -> np.ndarray:
    rho_reduced = rparams_to_tb(rho_params, keys, model._ndof)
    rho_new, _, mu, info = model.density_matrix(
        rho_reduced,
        keys=keys,
        mu_guess=state["mu"],
        max_mu_iterations=max_mu_iterations,
        max_subdivisions=max_subdivisions,
        rule=rule,
        batch_size=batch_size,
    )
    rho_new_reduced = {key: rho_new[key] for key in keys}
    residual = np.array(tb_to_rparams(rho_new_reduced) - rho_params, dtype=float).real

    state["mu"] = mu
    state["rho"] = rho_new_reduced
    state["info"] = info
    state["residual_norm"] = float(np.linalg.norm(residual))
    state["total_charge_integration_calls"] += info.charge_integration_calls
    state["total_density_integration_calls"] += info.density_integration_calls
    state["total_kernel_evals"] += info.n_kernel_evals
    state["total_evaluator_evals"] += info.n_evaluator_evals

    if debug:
        message = (
            f"mu={mu:.6g}, dN/dmu={info.dcharge_dmu:.6g}, "
            f"residual={state['residual_norm']:.6g}"
        )
        sys.stdout.write("\r" + message)
        sys.stdout.flush()

    return residual


def _solve_linear(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    alpha: float,
    maxiter: int,
    scf_tol: float,
    callback,
    state: dict,
) -> np.ndarray:
    x = np.array(x0, copy=True)
    for iteration in range(1, maxiter + 1):
        residual = residual_fn(x)
        state["iterations"] = iteration
        if callback is not None:
            callback(iteration, residual, state)
        if np.linalg.norm(residual) <= scf_tol:
            return x
        x = x + alpha * residual
    raise NoConvergence(x)


def _solve_anderson(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    maxiter: int,
    scf_tol: float,
    mixing_kwargs: dict,
    callback,
    state: dict,
) -> np.ndarray:
    kwargs = {"M": 5, "w0": 0.01, "maxiter": maxiter, "f_tol": scf_tol}
    kwargs.update(mixing_kwargs)

    def scipy_callback(x, f):
        del x
        state["iterations"] += 1
        if callback is not None:
            callback(state["iterations"], f, state)

    with np.errstate(invalid="ignore"):
        return scipy.optimize.anderson(residual_fn, x0, callback=scipy_callback, **kwargs)


def solver(
    model: Model,
    mf_guess: _tb_type,
    *,
    mu_guess: float = 0.0,
    mixing: str = "anderson",
    mixing_kwargs: dict | None = None,
    scf_tol: float | None = None,
    max_scf_steps: int = 100,
    callback=None,
    debug: bool = False,
    return_info: bool = False,
    rule: str = "auto",
    batch_size: int | None = None,
    max_mu_iterations: int = 32,
    max_subdivisions: int | None = 10_000,
) -> _tb_type | tuple[_tb_type, SolverInfo]:
    """Solve for the self-consistent mean-field correction."""
    keys = list(model.h_int)
    scf_tol = model.scf_tol if scf_tol is None else scf_tol
    mixing_kwargs = {} if mixing_kwargs is None else dict(mixing_kwargs)

    rho_guess, _, mu, info = density_matrix(
        model.hamiltonian_from_meanfield(mf_guess),
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        charge_tol=model.charge_tol,
        density_atol=model.density_atol,
        density_rtol=model.density_rtol,
        mu_guess=mu_guess,
        mu_xtol=model.mu_xtol,
        max_mu_iterations=max_mu_iterations,
        max_subdivisions=max_subdivisions,
        rule=rule,
        batch_size=batch_size,
    )
    rho_guess_reduced = {key: rho_guess[key] for key in keys}
    rho_params0 = tb_to_rparams(rho_guess_reduced)

    state = _scf_state()
    state["mu"] = mu
    state["rho"] = rho_guess_reduced
    state["info"] = info
    state["total_charge_integration_calls"] += info.charge_integration_calls
    state["total_density_integration_calls"] += info.density_integration_calls
    state["total_kernel_evals"] += info.n_kernel_evals
    state["total_evaluator_evals"] += info.n_evaluator_evals

    residual_fn = lambda params: _residual(
        params,
        model=model,
        keys=keys,
        state=state,
        debug=debug,
        rule=rule,
        batch_size=batch_size,
        max_mu_iterations=max_mu_iterations,
        max_subdivisions=max_subdivisions,
    )

    if mixing == "anderson":
        result_params = _solve_anderson(
            residual_fn,
            rho_params0,
            maxiter=max_scf_steps,
            scf_tol=scf_tol,
            mixing_kwargs=mixing_kwargs,
            callback=callback,
            state=state,
        )
    elif mixing == "linear":
        alpha = float(mixing_kwargs.pop("alpha", 0.5))
        result_params = _solve_linear(
            residual_fn,
            rho_params0,
            alpha=alpha,
            maxiter=max_scf_steps,
            scf_tol=scf_tol,
            callback=callback,
            state=state,
        )
    else:
        raise ValueError("mixing must be either 'anderson' or 'linear'")

    rho_result = rparams_to_tb(result_params, keys, model._ndof)
    rho_final, _, mu_final, info_final = model.density_matrix(
        rho_result,
        keys=keys,
        mu_guess=state["mu"],
        max_mu_iterations=max_mu_iterations,
        max_subdivisions=max_subdivisions,
        rule=rule,
        batch_size=batch_size,
    )
    rho_reduced_final = {key: rho_final[key] for key in keys}
    residual_norm = float(np.linalg.norm(tb_to_rparams(rho_reduced_final) - result_params))
    mf_result = meanfield(rho_reduced_final, model.h_int)
    tb_result = dict(mf_result)
    tb_result[model._local_key] = tb_result.get(
        model._local_key, np.zeros((model._ndof, model._ndof), dtype=complex)
    ) - mu_final * np.eye(model._ndof)

    solver_info = SolverInfo(
        mixing=mixing,
        iterations=max(1, state["iterations"]),
        residual_norm=residual_norm,
        mu=mu_final,
        total_charge_integration_calls=state["total_charge_integration_calls"]
        + info_final.charge_integration_calls,
        total_density_integration_calls=state["total_density_integration_calls"]
        + info_final.density_integration_calls,
        total_kernel_evals=state["total_kernel_evals"] + info_final.n_kernel_evals,
        total_evaluator_evals=state["total_evaluator_evals"] + info_final.n_evaluator_evals,
        last_density_info=info_final,
    )

    if debug:
        sys.stdout.write("\n")
    if return_info:
        return tb_result, solver_info
    return tb_result
