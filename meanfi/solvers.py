import sys
from functools import partial
import numpy as np
import scipy
from typing import Optional, Callable

from meanfi.params.rparams import rparams_to_tb, tb_to_rparams
from meanfi.tb.tb import add_tb, _tb_type
from meanfi.observables import expectation_value
from meanfi.mf import density_matrix, meanfield
from meanfi.model import Model
from meanfi.tb.utils import fermi_energy


def cost_mf(mf_param: np.ndarray, model: Model, nk: int = 20) -> np.ndarray:
    """Defines the cost function for root solver.

    The cost function is the difference between the computed and inputted mean-field.

    Parameters
    ----------
    mf_param :
        1D real array that parametrises the mean-field correction.
    Model :
        Interacting tight-binding problem definition.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        1D real array that is the difference between the computed and inputted mean-field
        parametrisations
    """
    shape = model._ndof
    mf = rparams_to_tb(mf_param, list(model.h_int), shape)
    mf_new = model.mfield(mf, nk=nk)
    mf_params_new = tb_to_rparams(mf_new)
    return mf_params_new - mf_param


def cost_density(rho_params_and_mu: np.ndarray, model: Model, debug: bool = False, bound_tol: float = 1e-1, save_history: bool = False) -> np.ndarray:
    """Defines the cost function for root solver.

    The cost function is the difference between the computed and inputted density matrix
    reduced to the hoppings only present in the h_int.

    Parameters
    ----------
    rho_params_and_mu :
        1D real array that parametrises the density matrix reduced to the
        hoppings (keys) present in h_int.
    Model :
        Interacting tight-binding problem definition.
    debug :
        Print debug information.
    bound_tol :
        Tolerance for the bounds of the filling.
    save_history :
        Save the history of the cost
    

    Returns
    -------
    :
        1D real array that is the difference between the computed and inputted
        density matrix parametrisations reduced to the hoppings present in h_int.
    """
    shape = model._ndof
    rho_params = rho_params_and_mu[:-1]
    mu = rho_params_and_mu[-1]
    rho_reduced = rparams_to_tb(rho_params, list(model.h_int), shape)
    keys = list(model.h_int)
    rho_new, E_min, E_max = model.density_matrix(rho_reduced, mu=mu, keys=keys)
    rho_reduced_new = {key: rho_new[key] for key in model.h_int}
    rho_params_new = tb_to_rparams(rho_reduced_new)
    n_operator = {model._local_key : np.eye(model._ndof)}
    charge = np.real(expectation_value(n_operator, rho_new))
    occupation_diff = np.real(charge - model.filling)
    
    added_cost = 0
    if charge > shape - bound_tol: 
        added_cost = mu - E_max
    if charge < bound_tol:
        added_cost = mu - E_min

    cost = np.array([*(rho_params_new - rho_params), occupation_diff + added_cost], dtype=float).real
    if save_history:
        cost_density.history.append((cost, rho_params_and_mu))

    if debug:
        message = f"Excess filling: {occupation_diff}, Chemical Potential: {mu}, Cost function: {np.linalg.norm(cost)}"
        sys.stdout.write('\r' + message)
        sys.stdout.flush()
    return cost


def solver_mf(
    model: Model,
    mf_guess: np.ndarray,
    nk: int = 20,
    optimizer: Optional[Callable] = scipy.optimize.anderson,
    optimizer_kwargs: Optional[dict[str, str]] = {"M": 0},
) -> _tb_type:
    """Solve for the mean-field correction through self-consistent root finding
    by finding the mean-field correction fixed point.

    Parameters
    ----------
    model :
        Interacting tight-binding problem definition.
    mf_guess :
        The initial guess for the mean-field correction in the tight-binding dictionary format.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.
    optimizer :
        The solver used to solve the fixed point iteration.
        Default uses `scipy.optimize.anderson`.
    optimizer_kwargs :
        The keyword arguments to pass to the optimizer.

    Returns
    -------
    :
        Mean-field correction solution in the tight-binding dictionary format.
    """
    shape = model._ndof
    mf_params = tb_to_rparams(mf_guess)
    f = partial(cost_mf, model=model, nk=nk)
    result = rparams_to_tb(
        optimizer(f, mf_params, **optimizer_kwargs), list(model.h_int), shape
    )
    fermi = fermi_energy(add_tb(model.h_0, result), model.filling, nk=nk)
    return add_tb(result, {model._local_key: -fermi * np.eye(model._ndof)})


def solver_density(
    model: Model,
    mf_guess: _tb_type,
    mu_guess: float, 
    optimizer: Optional[Callable] = scipy.optimize.anderson,
    optimizer_kwargs: Optional[dict[str, str]] = {"M": 0, "line_search": "wolfe"},
    debug: bool = False,
    optimizer_return = False,
    callback = None,
    save_history = False,
) -> _tb_type:
    """Solve for the mean-field correction through self-consistent root finding
    by finding the density matrix fixed point.

    Parameters
    ----------
    model :
        Interacting tight-binding problem definition.
    mf_guess :
        The initial guess for the mean-field correction in the tight-binding dictionary format.
    mu_guess :
        The initial guess for the chemical potential.
    optimizer :
        The solver used to solve the fixed point iteration.
        Default uses `scipy.optimize.anderson`.
    optimizer_kwargs :
        The keyword arguments to pass to the optimizer.
    debug :
        Print debug information.
    optimizer_return :
        Return the optimizer result.
    callback :
        Callback function to be called after each iteration.
    save_history :
        Save the history of the cost function.
    Returns
    -------
    :
        Mean-field correction solution in the tight-binding dictionary format.
    """
    shape = model._ndof
    keys = list(model.h_int)
    rho_guess = density_matrix(
        add_tb(model.h_0, mf_guess), mu=mu_guess, kT=model.kT, keys=keys, atol=model.atol
    )[0]
    rho_guess_reduced = {key: rho_guess[key] for key in model.h_int}

    rho_params = tb_to_rparams(rho_guess_reduced)
    rho_params_and_mu = np.concatenate([rho_params, [mu_guess]], dtype=float)
    if save_history:
        cost_density.history = []
    f = partial(cost_density, model=model, debug=debug, save_history=save_history)
    result = optimizer(f, rho_params_and_mu, callback=callback, **optimizer_kwargs)
    result_params = result.x
    rho_result = rparams_to_tb(
        result_params[:-1], list(model.h_int), shape
    )
    mf_result = meanfield(rho_result, model.h_int)

    tb_result = add_tb(mf_result, {model._local_key: -result_params[-1] * np.eye(model._ndof)})
    if optimizer_return:
        return tb_result, result
    return tb_result

solver = solver_density
