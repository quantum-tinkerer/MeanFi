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
        F_ij = np.einsum("kie,kje->kij", occ_vecs_flat, occ_vecs_flat.conj())
        reshape_order = [nk for i in range(dim)]
        reshape_order.extend([norbs, norbs])
        F = F_ij.reshape(*reshape_order)
    else:
        unocc_vals = vals > E_F
        occ_vecs = vecs
        occ_vecs[:, unocc_vals] = 0

        # Outter products between eigenvectors
        F = occ_vecs @ occ_vecs.T.conj()

    return F


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


def compute_mf(vals, vecs, filling, H_int):
    """
    Compute mean-field correction at self-consistent loop.

    Parameters:
    -----------
    vals : nd-array
        Eigenvalues of current loop vals[k_x, ..., k_n, j].
    vecs : nd_array
        Eigenvectors of current loop vals[k_x, ..., k_n, i, j].
    H_int : nd-array
        Interaction matrix.
    filling: int
        Number of electrons per cell.

    Returns:
    --------
    mf : nd-array
        Meanf-field correction with same format as `H_int`.
    """
    dim = len(vals.shape) - 1

    nk = vals.shape[0]

    E_F = utils.get_fermi_energy(vals, filling)
    F = mean_field_F(vals=vals, vecs=vecs, E_F=E_F)

    if dim > 0:
        H0_int = H_int[*[0 for i in range(dim)]]
        rho = np.diag(np.average(F, axis=tuple([i for i in range(dim)])))
        exchange_mf = convolution(F, H_int) * nk ** (-dim)
        direct_mf = np.diag(np.einsum("i,ij->j", rho, H0_int))
    else:
        rho = np.diag(F)
        exchange_mf = F * H_int
        direct_mf = np.diag(np.einsum("i,ij->j", rho, H_int))
    return direct_mf - exchange_mf


def scf_loop(mf, H_int, filling, hamiltonians_0):
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
    tol : float
        Tolerance of meanf-field self-consistent loop.
    """
    # Generate the Hamiltonian
    hamiltonians = hamiltonians_0 + mf
    vals, vecs = np.linalg.eigh(hamiltonians)
    vecs = np.linalg.qr(vecs)[0]

    mf_new = compute_mf(vals=vals, vecs=vecs, filling=filling, H_int=H_int)

    diff = mf_new - mf

    return diff


def find_groundstate_ham(
    tb_model,
    int_model,
    filling,
    nk=10,
    tol=1e-5,
    guess=None,
    mixing=0.01,
    order=10,
    verbose=False,
    return_mf=False,
):
    """
    Self-consistent loop to find groundstate Hamiltonian.

    Parameters:
    -----------
    H_int: nd-array
        Interaction matrix H_int[kx, ky, ..., i, j] where i,j are cell indices.
    filling: int
        Number of electrons per cell.
    hamiltonians_0 : nd-array
        Non-interacting Hamiltonian. Same format as `H_int`.
    tol : float
        Tolerance of meanf-field self-consistent loop.
    guess : nd-array
        Initial guess. Same format as `H_int`.
    mixing : float
        Regularization parameter in Anderson optimization. Default: 0.5.
    order : int
        Number of previous solutions to retain. Default: 1.
    verbose : bool
        Verbose of Anderson optimization. Default: False.

    Returns:
    --------
    hamiltonian : nd-array
        Groundstate Hamiltonian with same format as `H_int`.
    """
    hamiltonians_0, ks = utils.kgrid_hamiltonian(nk, tb_model, return_ks=True)
    if guess is None:
        guess = utils.generate_guess(nk, tb_model, int_model, scale=np.max(np.abs(np.array([*int_model.values()]))))
    fun = partial(
        scf_loop,
        hamiltonians_0=hamiltonians_0,
        H_int=utils.kgrid_hamiltonian(nk, int_model),
        filling=filling,
    )
    mf = anderson(fun, guess, f_tol=tol, w0=mixing, M=order, verbose=verbose)
    if return_mf:
        return utils.hk2tb_model(hamiltonians_0 + mf, tb_model, int_model, ks), mf
    else:
        return utils.hk2tb_model(hamiltonians_0 + mf, tb_model, int_model, ks)
