from functools import partial

import numpy as np
import scipy

from codes.params.rparams import rparams_to_tb, tb_to_rparams
from codes.tb.tb import add_tb
from codes.tb.utils import calculate_fermi_energy


def cost(mf_param, Model, nk=100):
    """Define the cost function for fixed point iteration.

    The cost function is the difference between the input mean-field real space
    parametrisation and a new mean-field.

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
    mf_tb = rparams_to_tb(mf_param, list(Model.h_int), shape)
    mf_tb_new = Model.mfield(mf_tb, nk=nk)
    mf_params_new = tb_to_rparams(mf_tb_new)
    return mf_params_new - mf_param


def solver(
    Model, mf_guess, nk=100, optimizer=scipy.optimize.anderson, optimizer_kwargs={}
):
    """Solve the mean-field self-consistent equation.

    Parameters
    ----------
    Model : Model
        The model object.
    mf_guess : numpy.array
        The initial guess for the mean-field tight-binding model.
    nk : int, optional
        The number of k-points to use in the grid. The default is 100. In the
        0-dimensional case, this parameter is ignored.
    optimizer : scipy.optimize, optional
        The optimizer to use to solve for fixed-points. The default is
        scipy.optimize.anderson.
    optimizer_kwargs : dict, optional
        The keyword arguments to pass to the optimizer. The default is {}.

    Returns
    -------
    result : numpy.array
        The mean-field tight-binding model.
    """
    shape = Model._size
    mf_params = tb_to_rparams(mf_guess)
    f = partial(cost, Model=Model, nk=nk)
    result = rparams_to_tb(
        optimizer(f, mf_params, **optimizer_kwargs), list(Model.h_int), shape
    )
    fermi = calculate_fermi_energy(add_tb(Model.h_0, result), Model.filling, nk=nk)
    return add_tb(result, {Model._local_key: -fermi * np.eye(Model._size)})
