from scipy.signal import convolve2d
import numpy as np
import utils
from functools import partial
from scipy.optimize import anderson, minimize


def mean_field_F(vals, vecs, E_F, nk):
    unocc_vals = vals > E_F

    def mf_generator(i, j):
        occ_vecs = vecs[i, j]
        occ_vecs[:, unocc_vals[i, j]] = 0
        F_ij = occ_vecs @ occ_vecs.conj().T
        return F_ij

    F = np.array([[mf_generator(i, j) for i in range(nk)] for j in range(nk)])
    return F


def convolution(M1, M2):
    cell_size = M2.shape[-1]
    V_output = np.array(
        [
            [
                convolve2d(M1[:, :, i, j], M2[:, :, i, j], boundary="wrap", mode="same")
                for i in range(cell_size)
            ]
            for j in range(cell_size)
        ]
    )
    V_output = np.transpose(V_output, axes=(2, 3, 0, 1))
    return V_output


def compute_mf(vals, vecs, filling, nk, H_int):
    H0_int = H_int[0, 0]
    E_F = utils.get_fermi_energy(vals, filling)
    F = mean_field_F(vals, vecs, E_F=E_F, nk=nk)
    rho = np.diag(np.average(F, axis=(0, 1)))
    exchange_mf = convolution(F, H_int) * nk ** (-2)
    direct_mf = np.diag(np.einsum("i,ij->j", rho, H0_int))

    return direct_mf - exchange_mf


def dm(mf0, mf):
    return np.mean(np.abs(mf - mf0))


def scf_loop(mf, H_int, nk, filling, hamiltonians_0, tol):
    if np.linalg.norm(mf) < tol:
        return 0
    # Generate the Hamiltonian
    hamiltonians = hamiltonians_0 + mf
    vals, vecs = np.linalg.eigh(hamiltonians)
    vecs = np.linalg.qr(vecs)[0]

    mf_new = compute_mf(vals=vals, vecs=vecs, filling=filling, nk=nk, H_int=H_int)

    diff = mf_new - mf

    if np.linalg.norm(mf_new) < tol:
        return 0
    else:
        return diff


def find_groundstate_ham(
    H_int, nk, filling, hamiltonians_0, tol, guess, mixing=0.5, order=1
):
    fun = partial(
        scf_loop,
        H_int=H_int,
        nk=nk,
        filling=filling,
        hamiltonians_0=hamiltonians_0,
        tol=tol,
    )
    mf = anderson(fun, guess, f_tol=tol, w0=mixing, M=order)
    return hamiltonians_0 + mf
