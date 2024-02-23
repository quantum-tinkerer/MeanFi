from . import utils
from functools import partial
import scipy

def kspace_costs(mfFlatReal, BaseMfModel):
    """
    Cost function for the mean-field model in k-space.

    Parameters
    ----------
    mfFlatReal : array_like
        Mean-field correction to the non-interacting hamiltonian in k-space.
        Converted from complex to stacked real array and flattened.
    BaseMfModel : object
        Mean-field model.
    
    Returns
    -------
    array_like
        Difference between the updated mean-field correction and the input.
    
    Notes
    -----
    In general, the cost function does the following steps:
    1. Does something with the input.
    2. Calls the meanField and densityMatrix methods of the BaseMfModel.
    3. Calculates output based on 2.

    In this case, the input and output is the mf correction in k-space. 
    Alternatively, it could also be a density matrix, or some real-space
    parametrisation of the mean-field. 
    """

    mf = utils.flat_to_matrix(
        utils.real_to_complex(mfFlatReal), BaseMfModel.H0_k.shape
        )
    rho = BaseMfModel.densityMatrix(mf)
    mfUpdated = BaseMfModel.meanField(rho)
    mfUpdatedFlatReal = utils.complex_to_real(utils.matrix_to_flat(mfUpdated))
    return mfUpdatedFlatReal - mfFlatReal

def kspace_solver(BaseMfModel, x0, optimizer=scipy.optimize.anderson, optimizer_kwargs={}):
    '''
    A solver needs to do the following things:
    1. Prepare input x0
    2. Prepare cost function 
    3. Run optimizer
    4. Prepare result
    5. Return result
    '''
    x0FlatReal = utils.complex_to_real(utils.matrix_to_flat(x0))
    f = partial(kspace_costs, BaseMfModel=BaseMfModel)
    return optimizer(f, x0FlatReal, **optimizer_kwargs)

def realSpace_cost(mfRealSpaceParams, BaseMfModel):
    """
    Cost function for the mean-field (mf) model in real space.
    The algorithm should go something like this:

    1. Convert the real-space parametrisation of the mf (mfRealSpaceParams)
    to mf in k-space.
    2. Generate density matrix rho via densityMatrix(mf, BaseMfModel)
    3. Generate mfUpdated via meanField(rho, BaseMfModel).
    4. Convert mfUpdated to real-space parametrisation.
    5. Return the difference the initial and updated parametrisations
    """
    return NotImplemented