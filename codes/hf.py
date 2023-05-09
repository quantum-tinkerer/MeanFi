from scipy.signal import convolve2d
import numpy as np
import utils

def mean_field_F(vals, vecs, E_F):
    N_ks = vecs.shape[0]
    unocc_vals = vals > E_F

    def mf_generator(i, j):
        occ_vecs = vecs[i, j]
        occ_vecs[:, unocc_vals[i, j]] = 0
        F_ij = occ_vecs @ occ_vecs.conj().T
        return F_ij

    F = np.array([[mf_generator(i, j) for i in range(N_ks)] for j in range(N_ks)])
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
    H0_int = H_int[0,0]
    E_F = utils.get_fermi_energy(vals, filling)
    F = mean_field_F(vals, vecs, E_F=E_F)
    rho = np.diag(np.average(F, axis=(0, 1)))
    exchange_mf = convolution(F, H_int) * nk ** (-2)
    direct_mf = np.diag(np.einsum("i,ij->j", rho, H0_int))

    return direct_mf - exchange_mf