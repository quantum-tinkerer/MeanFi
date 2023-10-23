import numpy as np
import kwant
from itertools import product

def get_fermi_energy(vals, filling):
    """
    Compute Fermi energy for a given filling factor.

    vals : nd-array
        Collection of eigenvalues on a grid.
    filling : int
        Number of electrons per cell.
    """
    norbs = vals.shape[-1]
    vals_flat = np.sort(vals.flatten())
    ne = len(vals_flat)
    ifermi = int(round(ne * filling / norbs))
    if ifermi >= ne:
        return vals_flat[-1]
    elif ifermi == 0:
        return vals_flat[0]
    else:
        fermi = (vals_flat[ifermi - 1] + vals_flat[ifermi]) / 2
        return fermi


def syst2hamiltonian(ks, syst, params={}, coordinate_names="xyz"):
    """
    Obtain Hamiltonian on k-point grid for a given `kwant` system.

    Paramters:
    ----------
    ks : 1D-array
        k-points (builds an uniform grid over all directions)
    syst : wrapped kwant system
        `kwant` system resulting from `kwant.wraparound.wraparound`
    params : dict
        System paramters.
    coordinate_names : 'str
        Same as `kwant.wraparound.wraparound`:
        The names of the coordinates along the symmetry directions of ‘builder’.

    Returns:
    --------
    ham : nd-array
        Hamiltnian matrix with format ham[k_x, ..., k_n, i, j], where i and j are internal degrees of freedom.
    """
    momenta = [
        "k_{}".format(coordinate_names[i])
        for i in range(len(syst._wrapped_symmetry.periods))
    ]

    def h_k(k):
        _k_dict = {}
        for i, k_n in enumerate(momenta):
            _k_dict[k_n] = k[i]
        return syst.hamiltonian_submatrix(params={**params, **_k_dict})

    k_pts = np.tile(ks, len(momenta)).reshape(len(momenta), len(ks))

    ham = []
    for k in product(*k_pts):
        ham.append(h_k(k))
    ham = np.array(ham)
    shape = (*np.repeat(k_pts.shape[1], k_pts.shape[0]), ham.shape[-1], ham.shape[-1])

    return ham.reshape(*shape)


def potential2hamiltonian(
    syst, lattice, func_onsite, func_hop, ks, params={}, max_neighbor=1
):
    """
    Construct an auxiliary `kwant` system to build Hamiltonian matrix.

    Parameters:
    -----------
    syst : `kwant.Builder`
        Non-interacting `kwant` system.
    lattice : `kwant.lattice`
        System lattice.
    func_onsite : function
        Onsite function.
    func_hop : function
        Hopping function.
    ks : 1D-array
        Set of k-points. Repeated for all directions.
    params : dict
        System parameters.
    max_neighbor : int
        Max nearest-neighbor order.

    Returns:
    --------
    H_int : nd-array
        Interaction matrix for the given onsite and hopping functions.
    """
    V = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    V[syst.sites()] = func_onsite
    for neighbors in range(max_neighbor):
        V[lattice.neighbors(neighbors + 1)] = func_hop
    wrapped_V = kwant.wraparound.wraparound(V).finalized()
    return syst2hamiltonian(ks=ks, syst=wrapped_V, params=params)


def assign_kdependence(
    ks, hopping_vecs, hopping_matrices
):
    """
    Computes Bloch matrix.

    ks : 1D-array
        Set of k-points. Repeated for all directions.
    hopping_vecs : 2D-array
        Hopping vectors.
    hopping_matrices : 3D-array
        Hopping matrices.

    Returns:
    --------
    bloch_matrix : nd-array
        Bloch matrix on a k-point grid.
    """
    ndof = hopping_matrices[0].shape[0]
    dim = len(hopping_vecs[0])
    nks = [len(ks) for i in range(dim)]
    bloch_matrix = np.zeros((nks + [ndof, ndof]), dtype=complex)
    kgrid = (
        np.asarray(np.meshgrid(*[ks for i in range(dim)]))
        .reshape(dim, -1)
        .T
    )

    for vec, matrix in zip(hopping_vecs, hopping_matrices):
        bloch_phase = np.exp(1j * np.dot(kgrid, vec)).reshape(nks + [1, 1])
        bloch_matrix += matrix.reshape([1 for i in range(dim)] + [ndof, ndof]) * bloch_phase

    return bloch_matrix


def generate_guess(nk, hopping_vecs, ndof, scale=0.1):
    """
    nk : int
        number of k points
    hopping_vecs : np.array
                hopping vectors as obtained from extract_hopping_vectors
    ndof : int
        number of degrees of freedom
    scale : float
            scale of the guess. If scale=1 then the guess is random around 0.5
            Smaller values of the guess significantly slows down convergence but
            does improve the result at phase instability points.

    Notes:
    -----
    Assumes that the desired max nearest neighbour distance is included in the hopping_vecs information.
    Creates a square grid by definition, might still want to change that
    """
    dim = len(hopping_vecs[0])
    all_rand_hermitians = []
    for n in hopping_vecs:
        amplitude = np.random.rand(ndof, ndof)
        phase = 2 * np.pi * np.random.rand(ndof, ndof)
        rand_hermitian = amplitude * np.exp(1j * phase)
        rand_hermitian += rand_hermitian.T.conj()
        rand_hermitian /= 2
        all_rand_hermitians.append(rand_hermitian)
    all_rand_hermitians = np.asarray(all_rand_hermitians)

    guess = assign_kdependence(nk, hopping_vecs, all_rand_hermitians)
    return guess * scale


def extract_hopping_vectors(builder):
    """
    Extract hopping vectors.

    Parameters:
    -----------
    builder : `kwant.Builder`

    Returns:
    --------
    hopping_vecs : 2d-array
        Hopping vectors stacked in a single array.
    """
    keep = None
    hopping_vecs = []
    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        b_dom = builder.symmetry.which(b)
        # Throw away part that is in the remaining translation direction, so we get
        # an element of 'sym' which is being wrapped
        b_dom = np.array([t for i, t in enumerate(b_dom) if i != keep])
        hopping_vecs.append(b_dom)
    return np.asarray(hopping_vecs)


def hk2hop(hk, hopping_vecs, ks):
    """
    Extract hopping matrices from Bloch Hamiltonian.

    Parameters:
    -----------
    hk : nd-array
        Bloch Hamiltonian matrix hk[k_x, ..., k_n, i, j]
    hopping_vecs : 2d-array
        Hopping vectors
    ks : 1D-array
        Set of k-points. Repeated for all directions.

    Returns:
    --------
    hopps : 3d-array
        Hopping matrices.
    """
    ndim = len(hk.shape) - 2
    dk = np.diff(ks)[0]
    nk = len(ks)
    k_pts = np.tile(ks, ndim).reshape(ndim, nk)
    k_grid = np.array(np.meshgrid(*k_pts))
    k_grid = k_grid.reshape(k_grid.shape[0], np.prod(k_grid.shape[1:]))
    # Can probably flatten this object to make einsum simpler
    hk = hk.reshape(np.prod(hk.shape[:ndim]), *hk.shape[-2:])

    hopps = (
        np.einsum(
            "ij,jkl->ikl",
            np.exp(1j * np.einsum("ij,jk->ik", hopping_vecs, k_grid)),
            hk,
        )
        * (dk / (2 * np.pi)) ** ndim
    )

    return hopps


def hk_densegrid(hk, ks, ks_dense, hopping_vecs):
    """
    Recomputes Hamiltonian on a denser k-point grid.

    Parameters:
    -----------
    hk : nd-array
        Coarse-grid Hamiltonian.
    ks : 1D-array
        Coarse-grid k-points.
    ks_dense : 1D-array
        Dense-grid k-points.
    hopping_vecs : 2d-array
        Hopping vectors.

    Returns:
    --------
    hk_dense : nd-array
        Bloch Hamiltonian computed on a dense grid.
    """
    hops = hk2hop(hk, hopping_vecs, ks)
    return assign_kdependence(ks_dense, hopping_vecs, hops)


def calc_gap(vals, E_F):
    """
    Compute gap.

    Parameters:
    -----------
    vals : nd-array
        Eigenvalues on a k-point grid.
    E_F : float
        Fermi energy.

    Returns:
    --------
    gap : float
        Indirect gap.
    """
    emax = np.max(vals[vals <= E_F])
    emin = np.min(vals[vals > E_F])
    return np.abs(emin - emax)
