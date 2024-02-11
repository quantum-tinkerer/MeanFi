from scipy.ndimage import convolve
import numpy as np
import codes.utils as utils

def density_matrix(vals, vecs, E_F):
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
    rho : array_like
        Density matrix rho=rho[kx, ky, ..., i, j] where i,j are cell indices.
    """
    norbs = vals.shape[-1]
    dim = len(vals.shape) - 1
    nk = vals.shape[0]

    if dim > 0:
        vals_flat = vals.reshape(-1, norbs)
        unocc_vals = vals_flat > E_F
        occ_vecs_flat = vecs.reshape(-1, norbs, norbs)
        occ_vecs_flat = np.transpose(occ_vecs_flat, axes=[0, 2, 1])
        occ_vecs_flat[unocc_vals, :] = 0
        occ_vecs_flat = np.transpose(occ_vecs_flat, axes=[0, 2, 1])

        # inner products between eigenvectors
        rho_ij = np.einsum("kie,kje->kij", occ_vecs_flat, occ_vecs_flat.conj())
        reshape_order = [nk for i in range(dim)]
        reshape_order.extend([norbs, norbs])
        rho = rho_ij.reshape(*reshape_order)
    else:
        unocc_vals = vals > E_F
        occ_vecs = vecs
        occ_vecs[:, unocc_vals] = 0

        # Outter products between eigenvectors
        rho = occ_vecs @ occ_vecs.T.conj()

    return rho


def convolution(M1, M2):
    """
    N-dimensional convolution.

    M1 : nd-array
    M2 : nd-array

    Returns:
    --------
    V_output : nd-array
        Discrete linear convolution of M1 with M2.
    """
    cell_size = M2.shape[-1]
    dim = len(M2.shape) - 2

    V_output = np.array(
        [
            [
                convolve(M1[..., i, j], M2[..., i, j], mode="wrap")
                for i in range(cell_size)
            ]
            for j in range(cell_size)
        ]
    )

    axes_order = np.roll(np.arange(dim + 2), shift=dim)
    V_output = np.transpose(V_output, axes=axes_order)
    return V_output


def compute_mf(rho, H_int):
    """
    Compute mean-field correction at self-consistent loop.

    Parameters:
    -----------
    rho : nd_array
        Density matrix.
    H_int : nd-array
        Interaction matrix.

    Returns:
    --------
    mf : nd-array
        Meanf-field correction with same format as `H_int`.
    """
    
    nk = rho.shape[0]
    dim = len(rho.shape) - 2
    
    if dim > 0:
        H0_int = H_int[*([0]*dim)]
        rho0 = rho[*([0]*dim)]
        local_density = np.diag(np.average(rho, axis=tuple([i for i in range(dim)])))
        exchange_mf = convolution(rho, H_int) * nk ** (-dim)
        direct_mf = np.diag(np.einsum("i,ij->j", local_density, H0_int))
        dc_energy_direct = np.einsum("i, j, ij->", local_density, local_density, H0_int)
        dc_energy_cross = np.einsum("ij, ji, ij->", H0_int, rho0, rho0)
        dc_energy = 2 * dc_energy_direct - dc_energy_cross
    else:
        local_density = np.diag(rho)
        exchange_mf = rho * H_int
        direct_mf = np.diag(np.einsum("i,ij->j", local_density, H_int))
        dc_energy_direct = np.diag(np.einsum("ij, i, j->", H_int, local_density, local_density))
        dc_energy_cross = np.diag(np.einsum("ij, ij, ji->", H_int, rho, rho))
        dc_energy = 2 * dc_energy_direct - dc_energy_cross
    return direct_mf - exchange_mf - dc_energy

def total_energy(h, rho):
    """
    Compute total energy.

    Paramters:
    ----------
    h : nd-array
        Hamiltonian.
    rho : nd-array
        Density matrix.

    Returns:
    --------
    total_energy : float
        System total energy computed as tr[h@rho].
    """
    return np.sum(np.trace(h @ rho, axis1=-1, axis2=-2)).real / np.prod(rho.shape[:-2])

def updated_matrices(mf_k, model):
    """
    Self-consistent loop.

    Parameters:
    -----------
    mf : nd-array
        Mean-field correction. Same format as the initial guess.
    H_int : nd-array
        Interaction matrix.
    filling: int
        Number of electrons per cell.
    hamiltonians_0 : nd-array
        Non-interacting Hamiltonian. Same format as `H_int`.

    Returns:
    --------
    mf_new : nd-array
        Updated mean-field solution.
    """
    # Generate the Hamiltonian
    hamiltonians = model.hamiltonians_0 + mf_k
    vals, vecs = np.linalg.eigh(hamiltonians)
    vecs = np.linalg.qr(vecs)[0]
    E_F = utils.get_fermi_energy(vals, model.filling)
    rho = density_matrix(vals=vals, vecs=vecs, E_F=E_F)
    return rho, compute_mf(
        rho=rho,
        H_int=model.H_int) - E_F * np.eye(model.hamiltonians_0.shape[-1])

