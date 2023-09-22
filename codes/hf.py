from scipy.ndimage import convolve
import numpy as np
import codes.utils as utils
from functools import partial
from scipy.optimize import anderson


def mean_field_F(vals, vecs, E_F):
    """
    Returns the mean field F_ij(k) = <psi_i(k)|psi_j(k)> for all k-points and
    eigenvectors below the Fermi level.
    
    Parameters
    ----------
    vals : array_like
        eigenvalues of the Hamiltonian
    vecs : array_like
        eigenvectors of the Hamiltonian
    E_F : float
        Fermi level
    
    Returns
    -------
    F : array_like
        mean field F[kx, ky, ..., i, j] where i,j are cell indices.
    """
    cell_size = vals.shape[-1]
    dim = len(vals.shape)-1
    nk = vals.shape[0] # not sure about this, since it forces a square grid

    vals_flat = vals.reshape(-1, cell_size)
    unocc_vals = vals_flat > E_F
    occ_vecs_flat = vecs.reshape(-1, cell_size, cell_size)
    occ_vecs_flat = np.transpose(occ_vecs_flat, axes=[0, 2, 1])
    occ_vecs_flat[unocc_vals, :] = 0
    occ_vecs_flat = np.transpose(occ_vecs_flat, axes=[0, 2, 1])
    
    # inner products between eigenvectors
    F_ij = np.einsum("kie,kje->kij", occ_vecs_flat, occ_vecs_flat.conj())

    reshape_order = [nk for i in range(dim)]
    reshape_order.extend([cell_size, cell_size])
    F = F_ij.reshape(*reshape_order)
    return F


def convolution(M1, M2):
    cell_size = M2.shape[-1]
    dim = len(M2.shape)-2
    
    V_output = np.array(
        [
            [
                convolve(M1[..., i, j], M2[..., i, j], mode="wrap")
                for i in range(cell_size)
            ]
            for j in range(cell_size)
        ]
    )

    axes_order = np.roll(np.arange(dim+2), shift=dim)
    V_output = np.transpose(V_output, axes=axes_order)
    return V_output


def compute_mf(vals, vecs, filling, H_int):
    dim = len(vals.shape)-1
    nk = vals.shape[0] 

    H0_int = H_int[*[0 for i in range(dim)]] # note the k-grid starts at k_x = k_y = 0
    E_F = utils.get_fermi_energy(vals, filling)
    F = mean_field_F(vals, vecs, E_F=E_F)
    rho = np.diag(np.average(F, axis=tuple([i for i in range(dim)])))
    exchange_mf = convolution(F, H_int) * nk ** (-dim)
    direct_mf = np.diag(np.einsum("i,ij->j", rho, H0_int))
    return direct_mf - exchange_mf


def scf_loop(mf, H_int, filling, hamiltonians_0):
    # Generate the Hamiltonian
    hamiltonians = hamiltonians_0 + mf
    vals, vecs = np.linalg.eigh(hamiltonians)
    vecs = np.linalg.qr(vecs)[0]
    mf_new = compute_mf(vals=vals, vecs=vecs, filling=filling, H_int=H_int)
    return np.array(np.abs(mf_new - mf), dtype=complex)


def find_groundstate_ham(
    H_int, filling, hamiltonians_0, tol, guess, mixing=0.5, order=1, verbose=False
):
    fun = partial(
        scf_loop,
        H_int=H_int,
        filling=filling,
        hamiltonians_0=hamiltonians_0
    )
    mf = anderson(fun, guess, f_rtol=tol, w0=mixing, M=order, verbose=verbose)
    return hamiltonians_0 + mf
