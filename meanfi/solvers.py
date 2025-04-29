# %%
from functools import partial
import numpy as np
import scipy
from typing import Optional, Callable

from params.rparams import (
    params_to_tb,
    tb_to_params,
    projection_to_tb,
    tb_to_projection,
    flatten_projection,
    unflatten_projection,
)
from tb.tb import add_tb, _tb_type
from tb.transforms import ham_fam_to_ort_basis
from mf import density_matrix, meanfield
from model import Model
from tb.utils import fermi_energy, symm_guess_mf


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
    mf = params_to_tb(mf_param, list(model.h_int), shape)
    mf_new = model.mfield(mf, nk=nk)
    mf_params_new = tb_to_params(mf_new)
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
    rho_reduced = params_to_tb(rho_params, list(model.h_int), shape)
    rho_new = model.density_matrix(rho_reduced, nk=nk)
    rho_reduced_new = {key: rho_new[key] for key in model.h_int}
    rho_params_new = tb_to_params(rho_reduced_new)
    return rho_params_new - rho_params


def cost_density_symmetric(
    rho_params: dict, model: Model, ham_basis: dict, nk: int = 20
) -> np.ndarray:
    """Defines the cost function for root solver.

    The cost function is the difference between the computed and inputted density matrix
    reduced to the hoppings only present in the h_int.

    Parameters
    ----------
    rho_params :
        1D real array that parametrises the density matrix reduced to the
        hoppings (keys) present in h_int.
    model :
        Interacting tight-binding problem definition.
    ham_basis :
        Dictionary containing basis matrices for the Hamiltonian.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        1D real array that is the difference between the computed and inputted
        density matrix parametrisations reduced to the hoppings present in h_int.
    """
    rho_params = unflatten_projection(rho_params, ham_basis)
    rho_reduced = projection_to_tb(rho_params, ham_basis)
    rho_new = model.density_matrix_iteration(rho_reduced, nk=nk)

    rho_reduced_new = {key: rho_new[key] for key in model.h_int}
    rho_params_new = tb_to_projection(rho_reduced_new, ham_basis)
    rho_params_new = flatten_projection(rho_params_new)

    return rho_params_new - flatten_projection(rho_params)


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
    mf_params = tb_to_params(mf_guess)

    f = partial(cost_mf, model=model, nk=nk)
    result = params_to_tb(
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

    rho_params = tb_to_params(rho_guess_reduced)
    f = partial(cost_density, model=model, nk=nk)
    rho_result = params_to_tb(
        optimizer(f, rho_params, **optimizer_kwargs), list(model.h_int), shape
    )
    mf_result = meanfield(rho_result, model.h_int)
    fermi = fermi_energy(add_tb(model.h_0, mf_result), model.filling, nk=nk)
    return add_tb(mf_result, {model._local_key: -fermi * np.eye(model._ndof)})


def solver_density_symmetric(
    model: Model,
    bloch_family: list[dict],
    guess: _tb_type = None,
    scale: float = 1,
    nk: int = 20,
    optimizer: Optional[Callable] = scipy.optimize.anderson,
    optimizer_kwargs: Optional[dict[str, str]] = {"M": 0, "line_search": "wolfe"},
) -> _tb_type:
    """Solve for the mean-field correction through self-consistent root finding
    by finding the density matrix fixed point.

    Parameters
    ----------
    model : Model
        Interacting tight-binding problem definition.
    bloch_family : list[dict]
        A list of `qsymm` `BlochModels` generated with `qsymm.bloch_family(..., bloch_model=True)` or `meanfi.tb.transforms.tb_to_ham_fam`.
    guess : _tb_type
        An initial guess for the mean-field correction.
    scale: float
        Scale of the random guess.
    nk : int
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
    if guess == None:
        mf_guess = symm_guess_mf(bloch_family, scale)
    else:
        mf_guess = guess

    ham_basis = ham_fam_to_ort_basis(bloch_family)

    rho_guess = density_matrix(
        add_tb(model.h_0, mf_guess), model.charge_op, model.target_charge, model.kT, nk
    )[0]

    rho_guess_reduced = {key: rho_guess[key] for key in model.h_int}

    rho_params = flatten_projection(tb_to_projection(rho_guess_reduced, ham_basis))

    f = partial(cost_density_symmetric, model=model, ham_basis=ham_basis, nk=nk)
    rho_result = projection_to_tb(
        unflatten_projection(optimizer(f, rho_params, **optimizer_kwargs), ham_basis),
        ham_basis,
    )

    # Not sure after this yet
    mf_result = meanfield(rho_result, model.h_int)
    # This should be changed.
    fermi = fermi_energy(add_tb(model.h_0, mf_result), model.target_charge, nk=nk)
    return add_tb(mf_result, {model._local_key: -fermi * np.eye(model._ndof)})


solver = solver_density

# %%
from tb.utils import superc_tb
import qsymm
from tb.transforms import tb_to_ham_fam

cutoff = 1
ndim = 1
ndof = 1

h_0 = superc_tb(cutoff, ndim, ndof)
h_int = superc_tb(cutoff, ndim, ndof)
tau_z = np.array([[1, 0], [0, -1]])
Q = np.kron(tau_z, np.eye(ndof))
target_Q = 0
kT = 1e-2
tau_x = np.kron(np.array([[0, 1], [1, 0]]), np.eye(ndof))

PHS = qsymm.particle_hole(ndim, tau_x)

symmetries = [PHS]

model = Model(h_0, h_int, Q, target_Q, kT)
hams = (h_0, h_int)

ndof = 2  # This is different because we generated the superconducting tb with ndof 1 per particle. (but there are two particles)
hoppings = list(
    h_int.keys()
)  # I believe this should still contain all hoppings, including h_0
ham_fam = tb_to_ham_fam(hoppings, ndof, symmetries)

print(h_0)
print(h_int)

print(solver_density_symmetric(model, ham_fam, nk=50))

# %%
