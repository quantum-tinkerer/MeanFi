import numpy as np
from . import utils
from .hf import updated_matrices
from functools import partial

def optimize(mf, cost_function, optimizer, optimizer_kwargs):
    _ = optimizer(
        cost_function,
        mf,
        **optimizer_kwargs
    )

def finite_system_cost(mf, model):
    shape = mf.shape
    mf = utils.flat_to_matrix(utils.real_to_complex(mf), shape)
    model.rho, model.mf_k = updated_matrices(mf_k=mf, model=model)
    delta_mf = model.mf_k - mf
    return utils.complex_to_real(utils.matrix_to_flat(delta_mf))

def finite_system_solver(model, optimizer, cost_function, optimizer_kwargs):
    """
    Real-space solver for finite systems.

    Parameters:
    -----------
    model : model.Model
        Physical model containting interacting and non-interacting Hamiltonian.
    optimizer : function
        Optimization function.
    optimizer_kwargs : dict
        Extra arguments passed to optimizer.
    """
    model.mf_k = model.guess[()]
    initial_mf = utils.complex_to_real(utils.matrix_to_flat(model.mf_k))
    partial_cost = partial(cost_function, model=model)
    optimize(initial_mf, partial_cost, optimizer, optimizer_kwargs)

def real_space_cost(mf, model, shape):
    mf = utils.flat_to_matrix(utils.real_to_complex(mf), shape)
    mf_dict = {}
    for i, key in enumerate(model.guess.keys()):
        mf_dict[key] = mf[i]
    mf = utils.kgrid_hamiltonian(
        nk=model.nk,
        hk=utils.model2hk(mf_dict),
        dim=model.dim,
        hermitian=False
    )
    model.rho, model.mf_k = updated_matrices(mf_k=mf, model=model)
    delta_mf = model.mf_k - mf
    delta_mf = utils.hk2tb_model(delta_mf, model.vectors, model.ks)
    delta_mf = np.array([*delta_mf.values()])
    return utils.complex_to_real(utils.matrix_to_flat(delta_mf))

def rspace_solver(model, optimizer, cost_function, optimizer_kwargs):
    """
    Real-space solver for infinite systems.

    Parameters:
    -----------
    model : model.Model
        Physical model containting interacting and non-interacting Hamiltonian.
    optimizer : function
        Optimization function.
    optimizer_kwargs : dict
        Extra arguments passed to optimizer.
    """
    model.kgrid_evaluation(nk=model.nk)
    initial_mf = np.array([*model.guess.values()])
    shape = initial_mf.shape
    initial_mf = utils.complex_to_real(utils.matrix_to_flat(initial_mf))
    partial_cost = partial(cost_function, model=model, shape=shape)
    optimize(initial_mf, partial_cost, optimizer, optimizer_kwargs)

def kspace_cost(mf, model):
    mf = utils.flat_to_matrix(utils.real_to_complex(mf), model.mf_k.shape)
    model.rho, model.mf_k = updated_matrices(mf_k=mf, model=model)
    delta_mf = model.mf_k - mf
    return utils.complex_to_real(utils.matrix_to_flat(delta_mf))

def kspace_solver(model, optimizer, cost_function, optimizer_kwargs):
    """
    k-space solver.

    Parameters:
    -----------
    model : model.Model
        Physical model containting interacting and non-interacting Hamiltonian.
    optimizer : function
        Optimization function.
    optimizer_kwargs : dict
        Extra arguments passed to optimizer.
    """
    model.kgrid_evaluation(nk=model.nk)
    initial_mf = model.mf_k
    initial_mf = utils.complex_to_real(utils.matrix_to_flat(initial_mf))
    partial_cost = partial(cost_function, model=model)
    optimize(initial_mf, partial_cost, optimizer, optimizer_kwargs)