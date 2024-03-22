from .params.rparams import mf2rParams, rParams2mf
import scipy
from functools import partial
from .tb.tb import addTb
import numpy as np


def cost(mf_param, Model, nK=100):
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
    nK : int, optional
        The number of k-points to use in the grid. The default is 100.
    """
    shape = Model._size
    mf_tb = rParams2mf(mf_param, list(Model.int_model), shape)
    mf_tb_new = Model.mfieldFFT(mf_tb, nK=nK)
    mf_params_new = mf2rParams(mf_tb_new)
    return mf_params_new - mf_param


def solver(
    Model, mf_guess, nK=100, optimizer=scipy.optimize.anderson, optimizer_kwargs={}
):
    """
    Solve the mean-field self-consistent equation.

    Parameters
    ----------
    Model : Model
        The model object.
    mf_guess : numpy.array
        The initial guess for the mean-field tight-binding model.
    nK : int, optional
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
    mf_params = mf2rParams(mf_guess)
    f = partial(cost, Model=Model, nK=nK)
    result = rParams2mf(
        optimizer(f, mf_params, **optimizer_kwargs), list(Model.int_model), shape
    )
    Model.calculateEF(nK=nK)
    localKey = tuple(np.zeros((Model._ndim,), dtype=int))
    return addTb(result, {localKey: -Model.EF * np.eye(shape)})
