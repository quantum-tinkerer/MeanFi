from functools import partial
import numpy as np
import scipy
from typing import Optional, Callable

from meanfi.params.rparams import (
    rparams_to_tb,
    tb_to_rparams,
    qparams_to_tb,
    tb_to_qparams,
)
from meanfi.tb.tb import add_tb, _tb_type
from meanfi.tb.transforms import tb_to_ham_fam, ham_fam_to_ort_basis
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


def cost_density(rho_params: np.ndarray, model: Model, nk: int = 20) -> np.ndarray:
    """Defines the cost function for root solver.

    The cost function is the difference between the computed and inputted density matrix
    reduced to the hoppings only present in the h_int.

    Parameters
    ----------
    rho_params :
        1D real array that parametrises the density matrix reduced to the
        hoppings (keys) present in h_int.
    Model :
        Interacting tight-binding problem definition.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        1D real array that is the difference between the computed and inputted
        density matrix parametrisations reduced to the hoppings present in h_int.
    """
    shape = model._ndof
    rho_reduced = rparams_to_tb(rho_params, list(model.h_int), shape)
    rho_new = model.density_matrix(rho_reduced, nk=nk)
    rho_reduced_new = {key: rho_new[key] for key in model.h_int}
    rho_params_new = tb_to_rparams(rho_reduced_new)
    return rho_params_new - rho_params


def cost_density_symmetric(
    rho_params: dict, model: Model, Q_basis: dict, nk: int = 20
) -> np.ndarray:
    """Defines the cost function for root solver.

    The cost function is the difference between the computed and inputted density matrix
    reduced to the hoppings only present in the h_int.

    Parameters
    ----------
    rho_params :
        1D real array that parametrises the density matrix reduced to the
        hoppings (keys) present in h_int.
    Model :
        Interacting tight-binding problem definition.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        1D real array that is the difference between the computed and inputted
        density matrix parametrisations reduced to the hoppings present in h_int.
    """
    rho_reduced = tb_to_qsymm.construct_ham(rho_params, Q_basis)
    rho_new = model.density_matrix(rho_reduced, nk=nk)

    rho_reduced_new = {key: rho_new[key] for key in model.h_int}
    rho_params_new = tb_to_qsymm.calculate_coefficients(rho_reduced_new, Q_basis)

    # Calculate the difference between the new params.
    # (0,): [a1, a2, a3]
    # Instead of dict, maybe do keys + values

    return rho_params_new - rho_params


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
    nk: int = 20,
    optimizer: Optional[Callable] = scipy.optimize.anderson,
    optimizer_kwargs: Optional[dict[str, str]] = {"M": 0, "line_search": "wolfe"},
) -> _tb_type:
    """Solve for the mean-field correction through self-consistent root finding
    by finding the density matrix fixed point.

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
    rho_guess = density_matrix(
        add_tb(model.h_0, mf_guess), filling=model.filling, nk=nk
    )[0]
    rho_guess_reduced = {key: rho_guess[key] for key in model.h_int}

    rho_params = tb_to_rparams(rho_guess_reduced)
    f = partial(cost_density, model=model, nk=nk)
    rho_result = rparams_to_tb(
        optimizer(f, rho_params, **optimizer_kwargs), list(model.h_int), shape
    )
    mf_result = meanfield(rho_result, model.h_int)
    fermi = fermi_energy(add_tb(model.h_0, mf_result), model.filling, nk=nk)
    return add_tb(mf_result, {model._local_key: -fermi * np.eye(model._ndof)})


def solver_density_symmetric(
    model: Model,
    guess: tuple = None,
    symmetries: tuple = None,  # We may want to make the symmetries an optional part of the model.
    nk: int = 20,
    optimizer: Optional[Callable] = scipy.optimize.anderson,
    optimizer_kwargs: Optional[dict[str, str]] = {"M": 0, "line_search": "wolfe"},
) -> _tb_type:
    """Solve for the mean-field correction through self-consistent root finding
    by finding the density matrix fixed point.

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

    # User should provide a guess that is a tuple of (bloch_family, coefficients).
    # If this is not provided, we generate these ourselves.
    if guess == None:
        ham_fam = tb_to_ham_fam((model.h_0, model.h_int), symmetries)
        ham_basis = ham_fam_to_ort_basis(ham_fam)

        scale = 5  # Arbitrary right now
        random_coeffs = {}
        # Can these be complex?
        for hopping in ham_basis:
            amplitude = scale * np.random.rand(len(ham_basis[hopping]))
            phase = 2 * np.pi * np.random.rand(len(ham_basis[hopping]))
            random_coeffs[hopping] = amplitude * np.exp(1j * phase)

        mf_guess = qparams_to_tb(random_coeffs, ham_basis)
    else:
        ham_fam, coefficients = guess
        ham_basis = ham_fam_to_ort_basis(ham_fam)

        mf_guess = qparams_to_tb(coefficients, ham_basis)

    # Not the right density matrix function.
    rho_guess = density_matrix(
        add_tb(model.h_0, mf_guess), filling=model.filling, nk=nk, kT=model.kT
    )[0]
    rho_guess_reduced = {key: rho_guess[key] for key in model.h_int}

    rho_params = tb_to_qparams(rho_guess_reduced, ham_basis)

    f = partial(cost_density_symmetric, model=model, Q_basis=ham_basis, nk=nk)
    rho_result = qparams_to_tb(optimizer(f, rho_params, **optimizer_kwargs), ham_basis)

    # Not sure after this yet
    mf_result = meanfield(rho_result, model.h_int)
    fermi = fermi_energy(add_tb(model.h_0, mf_result), model.filling, nk=nk)
    return add_tb(mf_result, {model._local_key: -fermi * np.eye(model._ndof)})


solver = solver_density
