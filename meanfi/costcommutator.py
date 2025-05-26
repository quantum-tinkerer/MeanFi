import numpy as np
from meanfi.tb.tb import _tb_type

def commutator(rho_tb: _tb_type, h_tb: _tb_type) -> _tb_type:
    """
    Compute the commutator of two dictionaries.
    
    Parameters
    ----------
    rho_tb :
        Tight-binding dictionary of the density matrix.
    h_tb :
        Tight-binding dictionary of the mean-field hamiltonian.
        
    Returns
    -------
    :
        Tight-binding dictionary of the commutator [rho, H]
    """
    shape = rho_tb[(0,)].shape[0]
    
    # Compute the commutator [rho, H]
    dim = len(list(h_tb.keys()))
    commut = {}
    for j in range(1, (dim + 1)//2):   # for blocks in the first row and the first column
        commut[(j,)] = 0
        commut[(-j,)] = 0
        for k in range(-(dim - 1)//2, (dim + 1)//2):
            if np.abs(j-k)<=(dim - 1)//2:
                commut[(j,)] += rho_tb[(k,)] @ h_tb[(j-k,)] - h_tb[(k,)] @ rho_tb[(j-k,)]
                commut[(-j,)] += rho_tb[(k-j,)] @ h_tb[(-k,)] - h_tb[(k-j,)] @ rho_tb[(-k,)]
    commut[(0,)] = 0             
    for k in range(-(dim - 1)//2, (dim + 1)//2):   # for blocks on the main diagonal
        commut[(0,)] += rho_tb[(k,)] @ h_tb[(-k,)] - h_tb[(k,)] @ rho_tb[(-k,)]
    return commut

    
def tb_to_matrix(tb: _tb_type) -> np.ndarray:
    """
    Convert a tight-binding dictionary to a matrix.
    
    Parameters
    ----------
    tb :
        Tight-binding dictionary.
    
    Returns
    -------
    :
        The matrix form parametrized by the dictionary.
    """
    dim = len(list(tb.keys()))
    matrix = tb[(0,)]
    for k in range(1, (dim + 1)//2):      # build the first row
        matrix = np.hstack((matrix, tb[(k,)]))
    for i in range(1, (dim + 1)//2):     # build all the other rows one at the time
        row = tb[(-i,)]
        for k in range(-i + 1, (dim + 1)//2 - i):
            row = np.hstack((row, tb[(k,)]))
        matrix = np.vstack((matrix, row))    # attach the row to the matrix
    return matrix
 