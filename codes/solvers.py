from codes.params.rparams import mf_to_rparams, rparams_to_mf
import scipy
from functools import partial
from codes.tb.tb import add_tb
import numpy as np


def cost(mf_param, Model, nk=100):
    """
    Define the cost function for fixed point iteration.
    The cost function is the difference between the input mean-field real space parametrisation
    and a new mean-field.

    Parameters
    ----------
    mf_param : numpy.array
        The mean-field real space parametrisation.
    Model : Model
        The model object.
    nk : int, optional
        The number of k-points to use in the grid. The default is 100.
    """
    shape = Model._size
    mf_tb = rparams_to_mf(mf_param, list(Model.h_int), shape)
    mf_tb_new = Model.mfield(mf_tb, nk=nk)
    mf_params_new = mf_to_rparams(mf_tb_new)
    return mf_params_new - mf_param


def solver(
    Model, mf_guess, nk=100, optimizer=scipy.optimize.anderson, optimizer_kwargs={}
):
    """
    Solve the mean-field self-consistent equation.

    Parameters
    ----------
    Model : Model
        The model object.
    mf_guess : numpy.array
        The initial guess for the mean-field tight-binding model.
    nk : int, optional
        The number of k-points to use in the grid. The default is 100.
    optimizer : scipy.optimize, optional
        The optimizer to use to solve for fixed-points. The default is scipy.optimize.anderson.
    optimizer_kwargs : dict, optional
        The keyword arguments to pass to the optimizer. The default is {}.

    Returns
    -------
    result : numpy.array
        The mean-field tight-binding model.
    """

    shape = Model._size
    mf_params = mf_to_rparams(mf_guess)
    f = partial(cost, Model=Model, nk=nk)
    result = rparams_to_mf(
        optimizer(f, mf_params, **optimizer_kwargs), list(Model.h_int), shape
    )
    Model.calculate_EF()
    local_key = tuple(np.zeros((Model._ndim,), dtype=int))
    return add_tb(result, {local_key: -Model.EF * np.eye(shape)})
