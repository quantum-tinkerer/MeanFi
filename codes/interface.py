from scipy import optimize
from . import utils, solvers
import numpy as np

def find_groundstate_ham(
    model,
    filling,
    nk=10,
    cutoff_Vk=0,
    solver=solvers.kspace_solver,
    cost_function=solvers.kspace_cost,
    optimizer=optimize.anderson,
    optimizer_kwargs={},
    return_mf=False,
    return_kspace=False
):
    """
    Self-consistent loop to find groundstate Hamiltonian.

    Parameters:
    -----------
    tb_model : dict
        Tight-binding model. Must have the following structure:
            - Keys are tuples for each hopping vector (in units of lattice vectors).
            - Values are hopping matrices.
    filling: int
        Number of electrons per cell.
    guess : nd-array
        Initial guess. Same format as `H_int`.
    return_mf : bool
        Returns mean-field result. Useful if wanted to reuse as guess in upcoming run.

    Returns:
    --------
    scf_model : dict
        Tight-binding model of Hartree-Fock solution.
    """
    model.nk=nk
    model.filling=filling
    if model.int_model is not None:
        model.vectors=[*model.int_model.keys()]
    else:
        model.vectors = utils.generate_vectors(cutoff_Vk, model.dim)
    if model.guess is None:
        model.random_guess(model.vectors)
    solver(model, optimizer, cost_function, optimizer_kwargs)
    model.vectors=[*model.vectors, *model.tb_model.keys()]
    assert np.allclose(model.mf_k - np.moveaxis(model.mf_k, -1, -2).conj(), 0, atol=1e-15)
    if return_kspace:
        return model.hamiltonians_0 + model.mf_k
    else:
        if model.dim > 0:
            scf_tb = utils.hk2tb_model(model.hamiltonians_0 + model.mf_k, model.vectors, model.ks)
            if return_mf:
                mf_tb = utils.hk2tb_model(model.mf_k, model.vectors, model.ks)
                return scf_tb, mf_tb
            else:
                return scf_tb
        else:
            if return_mf:
                return {() : model.hamiltonians_0 + model.mf_k}, {() : model.mf_k}
            else:
                return {() : model.hamiltonians_0 + model.mf_k}
