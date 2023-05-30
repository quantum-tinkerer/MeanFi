from scipy.ndimage import convolve
import numpy as np
import utils
from functools import partial
from scipy.optimize import anderson


def mean_field_F(vals, vecs, E_F):
    cell_size = vals.shape[-1]
    dim = len(vals.shape)-1
    nk = vals.shape[0]

    vals_flat = vals.reshape(-1, cell_size)
    unocc_vals = vals_flat > E_F
    occ_vecs_flat = vecs.reshape(-1, cell_size, cell_size)
    occ_vecs_flat[unocc_vals] = 0
    
    # inner products between eigenvectors
    F_ij = np.einsum("ijk,kli->ijl", occ_vecs_flat, occ_vecs_flat.conj().T)

    reshape_order = [nk for i in range(dim)]
    reshape_order.extend([cell_size, cell_size])
    F = F_ij.reshape(*reshape_order)
    return F


def convolution(M1, M2):
    cell_size = M2.shape[-1]
    dim = len(M2.shape)-2

    axes_order = [i for i in range(2, dim+2, 1)]
    axes_order.extend([0, 1])
    axes_order = tuple(axes_order)

    def T_kx_cell(M):
        # transpose the k with cell indices
        return np.transpose(M, axes=axes_order)
    
    V_output = np.array(
        [
            [
                convolve(T_kx_cell(M1)[i, j, ],T_kx_cell(M2)[i, j, ], mode="wrap")
                for i in range(cell_size)
            ]
            for j in range(cell_size)
        ]
    )
    print(axes_order)
    V_output = T_kx_cell(V_output)
    return V_output


def compute_mf(vals, vecs, filling, H_int):
    dim = len(vals.shape)-1
    nk = vals.shape[0] 

    H0_int = H_int[*[0 for i in range(dim)]] # note the k-grid starts at k_x = k_y = 0
    E_F = utils.get_fermi_energy(vals, filling)
    F = mean_field_F(vals, vecs, E_F=E_F)
    rho = np.diag(np.average(F, axis=tuple([i for i in range(dim)])))
    exchange_mf = convolution(F, H_int) * nk ** (-2)
    direct_mf = np.diag(np.einsum("i,ij->j", rho, H0_int))

    return direct_mf - exchange_mf


def scf_loop(mf, H_int, filling, hamiltonians_0, tol):
    if np.linalg.norm(mf) < tol:
        return 0
    # Generate the Hamiltonian
    hamiltonians = hamiltonians_0 + mf
    vals, vecs = np.linalg.eigh(hamiltonians)
    vecs = np.linalg.qr(vecs)[0]

    mf_new = compute_mf(vals=vals, vecs=vecs, filling=filling, H_int=H_int)

    diff = mf_new - mf

    if np.linalg.norm(mf_new) < tol:
        return 0
    else:
        return diff


def find_groundstate_ham(
    H_int, filling, hamiltonians_0, tol, guess, mixing=0.5, order=1
):
    fun = partial(
        scf_loop,
        H_int=H_int,
        filling=filling,
        hamiltonians_0=hamiltonians_0,
        tol=tol,
    )
    mf = anderson(fun, guess, f_tol=tol, w0=mixing, M=order)
    return hamiltonians_0 + mf
