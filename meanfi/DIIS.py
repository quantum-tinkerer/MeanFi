import numpy as np
import meanfi
from meanfi.model import Model
from meanfi.mf import density_matrix, meanfield
from meanfi.tb.tb import add_tb, _tb_type, scale_tb
from costcommutator import commutator, tb_to_matrix

def product(tb1: _tb_type, tb2: _tb_type) -> _tb_type:
    """
    Compute the product tb1 @ tb2 of two dictionaries.
    
    Parameters
    ----------
    tb1 :
        First tight-binding dictionary.
    tb2 :
        Second tight-binding dictionary.
        
    Returns
    -------
    :
        Product of the two tight-binding dictionaries.
    """
    shape = tb1[(0,)].shape[0]
    dim = len(list(tb1.keys()))
    product = {}
    for j in range(1, (dim + 1)//2):   # for blocks in the first row and the first column
        product[(j,)] = 0
        product[(-j,)] = 0
        for k in range(-(dim -1)//2, (dim + 1)//2):
            if np.abs(j-k)<=(dim - 1)//2:
                product[(j,)] += tb1[(k,)] @ tb2[(j-k,)]
                product[(-j,)] += tb1[(k-j,)] @ tb2[(-k,)]
    product[(0,)] = 0     
    for k in range(-(dim - 1)//2, (dim + 1)//2):   # for blocks on the main diagonal
        product[(0,)] += tb1[(k,)] @ tb2[(-k,)]
    return product


def dot(tb1: _tb_type, tb2: _tb_type) -> _tb_type:
    """
    Compute the dot product of two dictionaries, defined as Tr(tb1^dagger @ tb2).
    
    Parameters
    ----------
    tb1 :
        First tight-binding dictionary.
    tb2 :
        Second tight-binding dictionary.
        
    Returns
    -------
    :
        Dot product of the two tight-binding dictionaries.
    """
    tb1_dag = {tuple(-np.array(k)):np.conj(tb1[k].T) for k in tb1.keys()}
    prod = product(tb1_dag, tb2)
    return np.trace(prod[(0,)])


def diis_coeff(err_vec: _tb_type, coeff_matrix: np.ndarray, err_list: list) -> np.ndarray:
    """
    Compute the DIIS coefficients.
    
    Parameters
    ----------
    err_vec :
        Error vector dictionary at the current step of the cycle.
    coeff_matrix :
        Matrix of coefficients of the previous step.
    err_list :
        List of the error vectors of the previous step.
        
    Returns
    -------
    :
        DIIS coefficients, updated coefficients matrix, updated list of the error vectors.
    """
    err_list.append(err_vec)
    column = np.zeros(len(err_list), dtype='complex')
    for i in range(len(err_list)):
        column[i] = dot(err_list[i], err_list[-1])
    coeff_matrix = np.vstack((np.hstack((coeff_matrix, column[:-1, np.newaxis])), np.conj(column)))
    # Construction of the linear system (Ax=b)
    dim = coeff_matrix.shape[0]
    A = np.vstack((np.hstack((coeff_matrix, np.ones((dim,1)))), np.ones(dim+1)))
    A[-1, -1] = 0
    b = np.zeros(dim+1)
    b[-1] = 1
    solution = np.linalg.solve(A, b)
    return solution[:-1], coeff_matrix, err_list
    
        
def DIIS(model: Model, n: int, nk: int, tol: float, niter: int) -> _tb_type:
    """
    DIIS cycle.
    
    Parameters
    ----------
    model :
        Model that defines the interacting tight-binding problem.
    n :
        Number of error vectors of the previous steps used in the calculation.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension. 
    tol :
        Tollerance threshold.
    niter :
        Maximum number of iterations.
        
    Returns
    -------
    :
        Total meanf-field hamiltonian.
    """
    rho_0, fermi_energy = density_matrix(model.h_0, model.filling, nk)
    mf_correction = meanfield(rho_0, model.h_int)
    h_mf = add_tb(model.h_0, mf_correction)
    e0 = commutator(rho_0, h_mf)
    residual = dot(e0, e0)
    err_list = [e0]
    h_list = [h_mf]
    nstep = 0
    h_i = h_mf
    coeff_matrix = np.array([[residual]])
    while residual >= tol and nstep <= niter:
        rho_i, fermi_energy = density_matrix(h_i, model.filling, nk)
        mf_correction = meanfield(rho_i, model.h_int)
        h_mf = add_tb(add_tb(model.h_0, mf_correction), {model._local_key: -fermi_energy * np.eye(model._ndof)})
        h_list.append(h_mf)
        e_i = commutator(rho_i, h_mf)
        coeff, coeff_matrix, err_list = diis_coeff(e_i, coeff_matrix, err_list)
            
        # H_new = \sum_j c_j*h_j
        print('len coeff %i' %len(coeff))
        print('len err_list %i' %len(err_list))
        print('len h_list %i' %len(h_list))
        h_new = scale_tb(h_list[0], coeff[0])
        for j in range(1, len(h_list)):
            h_new = add_tb(h_new, scale_tb(h_list[j], coeff[j]))
            
        if len(err_list) > n:
            del err_list[0]
            del h_list[0]
            coeff_matrix = np.delete(coeff_matrix, (0), axis=0)
            coeff_matrix = np.delete(coeff_matrix, (0), axis=1)
            
        h_i = h_new
        residual = dot(e_i, e_i)    #alternatively residual = coeff_matrix[-1, -1]
        print('n_step = {}, residual = {}' .format(nstep, np.real(residual)))
        nstep += 1
    return h_i