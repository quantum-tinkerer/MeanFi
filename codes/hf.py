from scipy.ndimage import convolve
import numpy as np
import codes.utils as utils
from functools import partial
from scipy.optimize import anderson, minimize

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
    
    nk = rho.shape[0]
    dim = len(rho.shape) - 2
    
    if dim > 0:
        H0_int = H_int[*([0]*dim)]
        local_density = np.diag(np.average(rho, axis=tuple([i for i in range(dim)])))
        exchange_mf = convolution(rho, H_int) * nk ** (-dim)
        direct_mf = np.diag(np.einsum("i,ij->j", local_density, H0_int))
    else:
        local_density = np.diag(rho)
        exchange_mf = rho * H_int
        direct_mf = np.diag(np.einsum("i,ij->j", local_density, H_int))
    return direct_mf - exchange_mf

def total_energy(h, rho):
    return np.sum(np.trace(h @ rho, axis1=-1, axis2=-2))

class Model:

    def __init__(self, tb_model, int_model=None, Vk=None, guess=None):
        self.tb_model = tb_model
        self.hk = utils.model2hk(tb_model=tb_model)
        if int_model is not None:
            self.int_model = int_model
            self.Vk = utils.model2hk(tb_model=int_model)
        else:
            self.Vk = Vk
        self.dim = len([*tb_model.keys()][0])
        self.ndof = len([*tb_model.values()][0])
        self.guess = guess
            

    def random_guess(self, vectors):
        self.guess = utils.generate_guess(
            vectors=vectors,
            ndof=self.ndof,
            scale=1
        )

    def kgrid_evaluation(self, nk):
        self.hamiltonians_0, self.ks = utils.kgrid_hamiltonian(
            nk=nk,
            hk=self.hk,
            dim=self.dim,
            return_ks=True
        )
        self.H_int = utils.kgrid_hamiltonian(nk=nk, hk=self.Vk, dim=self.dim)
        self.mf_k = utils.kgrid_hamiltonian(
            nk=nk,
            hk=utils.model2hk(self.guess),
            dim=self.dim,
        )

    def flatten_mf(self):
        flat = self.mf_k.flatten()
        return flat[:len(flat)//2] + 1j * flat[len(flat)//2:]

    def reshape_mf(self, mf_flat):
        mf_flat = np.concatenate((np.real(mf_flat), np.imag(mf_flat)))
        return mf_flat.reshape(*self.hamiltonians_0.shape)

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
    return rho, compute_mf(rho=rho, H_int=model.H_int) - E_F * np.eye(model.hamiltonians_0.shape[-1])

def default_cost(mf, model):
    model.rho, model.mf_k = updated_matrices(mf_k=mf, model=model)
    model.energy = total_energy(h=model.hamiltonians_0 + model.mf_k, rho=model.rho)
    h = model.hamiltonians_0 + model.mf_k
    commutator = h@model.rho - model.rho@h
    n_herm = (mf - np.moveaxis(mf, -1, -2).conj())/2
    delta_mf = model.mf_k - mf
    quantities = np.array([commutator, delta_mf, n_herm])
    idx_max = np.argmax(np.linalg.norm(quantities.reshape(3, -1), axis=-1))
    return quantities[idx_max]

def find_groundstate_ham(
    model,
    cutoff_Vk,
    filling,
    nk=10,
    cost_function=default_cost,
    optimizer=anderson,
    optimizer_kwargs={},
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
    vectors = utils.generate_vectors(cutoff_Vk, model.dim)
    model.vectors=[*vectors, *model.tb_model.keys()]
    if model.guess is None:
        model.random_guess(model.vectors)
    model.kgrid_evaluation(nk=nk)
    fun = partial(
        cost_function,
        model=model
    )
    mf_k = optimizer(
        fun,
        model.mf_k,
        **optimizer_kwargs
    )
    assert np.allclose((mf_k - np.moveaxis(mf_k, -1, -2).conj())/2, 0, atol=1e-5)
    mf_k = (mf_k + np.moveaxis(mf_k, -1, -2).conj())/2
    return utils.hk2tb_model(model.hamiltonians_0 + mf_k, model.vectors, model.ks)
