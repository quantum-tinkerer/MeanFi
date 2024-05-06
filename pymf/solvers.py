from functools import partial
import numpy as np
import scipy
from typing import Optional, Callable

from pymf.params.rparams import rparams_to_tb, tb_to_rparams
from pymf.tb.tb import add_tb, tb_type
from pymf.model import Model
from pymf.tb.utils import calculate_fermi_energy


def cost(mf_param: np.ndarray, model: Model, nk: Optional[int] = 100) -> np.ndarray:
    """Define the cost function for fixed point iteration.

    The cost function is the difference between the input mean-field real space
    parametrisation and a new mean-field.

    Parameters
    ----------
    mf_param :
        1D real array that parametrises the mean-field tight-binding correction.
    Model :
        Object which defines the mean-field problem to solve.
    nk :
        The number of k-points to use in the grid.

    Returns
    -------
    :
        1D real array which contains the difference between the input mean-field
        and the new mean-field.
    """
    shape = model._size
    mf_tb = rparams_to_tb(mf_param, list(model.h_int), shape)
    mf_tb_new = model.mfield(mf_tb, nk=nk)
    mf_params_new = tb_to_rparams(mf_tb_new)
    return mf_params_new - mf_param


def solver(
    model: Model,
    mf_guess: np.ndarray,
    nk: Optional[int] = 100,
    optimizer: Optional[Callable] = scipy.optimize.anderson,
    optimizer_kwargs: Optional[dict[str, str]] = {},
) -> tb_type:
    """Solve the mean-field self-consistent equation.

    Parameters
    ----------
    model :
        The model object.
    mf_guess :
        The initial guess for the mean-field tight-binding model.
    nk :
        The number of k-points to use in the grid. The default is 100. In the
        0-dimensional case, this parameter is ignored.
    optimizer :
        The solver used to solve the fixed point iteration.
    optimizer_kwargs :
        The keyword arguments to pass to the optimizer.

    Returns
    -------
    :
        The mean-field tight-binding model.
    """
    shape = model._size
    mf_params = tb_to_rparams(mf_guess)
    f = partial(cost, model=model, nk=nk)
    result = rparams_to_tb(
        optimizer(f, mf_params, **optimizer_kwargs), list(model.h_int), shape
    )
    fermi = calculate_fermi_energy(add_tb(model.h_0, result), model.filling, nk=nk)
    return add_tb(result, {model._local_key: -fermi * np.eye(model._size)})
