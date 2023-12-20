from scipy.ndimage import convolve
import numpy as np
import codes.utils as utils
from scipy import optimize

def density_matrix(vals, vecs, E_F, dim):
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


def compute_mf(rho, H_int, dim):
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
    return np.sum(np.trace(h @ rho, axis1=-1, axis2=-2)).real

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
    rho = density_matrix(vals=vals, vecs=vecs, E_F=E_F, dim=model.dim)
    return rho, compute_mf(
        rho=rho,
        H_int=model.H_int,
        dim=model.dim) - E_F * np.eye(model.hamiltonians_0.shape[-1])

def finite_system_solver(model, optimizer, optimizer_kwargs):
    """
    Real-space solver for finite systems.

    Parameters:
    -----------
    model : model.Model
        Physical model containting interacting and non-interacting Hamiltonian.
    optimizer : function
        Optimization function.
    optimizer_kwargs : dict
        Extra arguments passed to optimizer.
    """
    mf = model.guess[()]
    shape = mf.shape

    def cost_function(mf):
        mf = utils.flat_to_matrix(utils.real_to_complex(mf), shape)
        model.rho, model.mf_k = updated_matrices(mf_k=mf, model=model)
        delta_mf = model.mf_k - mf
        return utils.complex_to_real(utils.matrix_to_flat(delta_mf))

    _ = optimizer(
        cost_function,
        utils.complex_to_real(utils.matrix_to_flat(mf)),
        **optimizer_kwargs
    )

def rspace_solver(model, optimizer, optimizer_kwargs):
    """
    Real-space solver for infinite systems.

    Parameters:
    -----------
    model : model.Model
        Physical model containting interacting and non-interacting Hamiltonian.
    optimizer : function
        Optimization function.
    optimizer_kwargs : dict
        Extra arguments passed to optimizer.
    """
    model.kgrid_evaluation(nk=model.nk)
    mf = np.array([*model.guess.values()])
    shape = mf.shape

    def cost_function(mf):
        mf = utils.flat_to_matrix(utils.real_to_complex(mf), shape)
        mf_dict = {}
        for i, key in enumerate(model.guess.keys()):
            mf_dict[key] = mf[i]
        mf = utils.kgrid_hamiltonian(
            nk=model.nk,
            hk=utils.model2hk(mf_dict),
            dim=model.dim,
            hermitian=False
        )
        model.rho, model.mf_k = updated_matrices(mf_k=mf, model=model)
        model.energy = total_energy(h=model.hamiltonians_0 + model.mf_k, rho=model.rho)
        delta_mf = model.mf_k - mf
        delta_mf = utils.hk2tb_model(delta_mf, model.vectors, model.ks)
        delta_mf = np.array([*delta_mf.values()])
        return utils.complex_to_real(utils.matrix_to_flat(delta_mf))

    _ = optimizer(
        cost_function,
        utils.complex_to_real(utils.matrix_to_flat(mf)),
        **optimizer_kwargs
    )


def kspace_solver(model, optimizer, optimizer_kwargs):
    """
    k-space solver.

    Parameters:
    -----------
    model : model.Model
        Physical model containting interacting and non-interacting Hamiltonian.
    optimizer : function
        Optimization function.
    optimizer_kwargs : dict
        Extra arguments passed to optimizer.
    """
    model.kgrid_evaluation(nk=model.nk)
    def cost_function(mf):
        mf = utils.flat_to_matrix(utils.real_to_complex(mf), model.mf_k.shape)
        model.rho, model.mf_k = updated_matrices(mf_k=mf, model=model)
        model.energy = total_energy(h=model.hamiltonians_0 + model.mf_k, rho=model.rho)
        delta_mf = model.mf_k - mf
        return utils.complex_to_real(utils.matrix_to_flat(delta_mf))

    _ = optimizer(
        cost_function,
        utils.complex_to_real(utils.matrix_to_flat(model.mf_k)),
        **optimizer_kwargs
    )

def find_groundstate_ham(
    model,
    cutoff_Vk,
    filling,
    nk=10,
    solver=kspace_solver,
    optimizer=optimize.anderson,
    optimizer_kwargs={'M':0, 'verbose': False},
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
    solver(model, optimizer, optimizer_kwargs)
    model.vectors=[*model.vectors, *model.tb_model.keys()]
    assert np.allclose((model.mf_k - np.moveaxis(model.mf_k, -1, -2).conj())/2, 0, atol=1e-15)
    if model.dim > 0:
        return utils.hk2tb_model(model.hamiltonians_0 + model.mf_k, model.vectors, model.ks)
    else:
        return {() : model.hamiltonians_0 + model.mf_k}
