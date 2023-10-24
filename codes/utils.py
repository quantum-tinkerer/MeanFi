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


def kwant2hk(syst, params={}, coordinate_names="xyz"):
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

    def bloch_ham(k):
        _k_dict = {}
        for i, k_n in enumerate(momenta):
            _k_dict[k_n] = k[i]
        return syst.hamiltonian_submatrix(params={**params, **_k_dict})

    return bloch_ham


def builder2tb_model(builder):
    from copy import copy

    builder = copy(bulk_graphene)

    tb_model = {}
    sites_list = [*builder.sites()]
    norbs_list = [site[0].norbs for site in builder.sites()]
    positions_list = [site[0].pos for site in builder.sites()]
    norbs_tot = sum(norbs_list)
    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        print(a.pos, b.pos)
        b_dom = builder.symmetry.which(b)
        b_fd = builder.symmetry.to_fd(b)
        atoms = np.array([sites_list.index(a), sites_list.index(b_fd)])
        row, col = [
            np.sum(norbs_list[: atoms[0]]) + range(norbs_list[atoms[0]]),
            np.sum(norbs_list[: atoms[1]]) + range(norbs_list[atoms[1]]),
        ]
        row, col = np.array([*product(row, col)]).T
        data = val.flatten()
        if tuple(b_dom) in tb_model:
            tb_model[tuple(b_dom)] += coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
        else:
            tb_model[tuple(b_dom)] = coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
    tb_model['norbs'] = norbs_list
    tb_model['positions'] = positions_list
    
    return tb_model


def dict2hk(tb_dict):
    """
    Build Bloch Hamiltonian.


    Returns:
    --------
    ham : function
        Evaluates the Hamiltonian at a give k-point
    """

    def bloch_ham(k):
        ham = sum(
            tb_dict[vector] * np.exp(1j * np.dot(k, np.array(vector)))
            for vector in tb_dict.keys()
        )
        # ham += ham.T.conj()
        return ham

    return bloch_ham


def kgrid_hamiltonian(nk, syst, params={}, return_ks=False):
    """
    Evaluates Hamiltonian on a k-point grid.

    Paramters:
    ----------
    nk : int
        Number of k-points along each direction.
    tb_dict : dictionary
        Must have the following structure:
            - Keys are tuples for each hopping vector (in units of lattice vectors).
            - Values are hopping matrices.

    Returns:
    --------
    ham : nd.array
        Hamiltonian evaluated on a k-point grid from k-points
        along each direction evaluated from zero to 2*pi.
        The indices are ordered as [k_1, ... , k_n, i, j], where
        `k_m` corresponding to the k-point element along each
        direction and `i` and `j` are the internal degrees of freedom.
    """
    if type(syst) == kwant.builder.FiniteSystem:
        try:
            dim = len(syst._wrapped_symmetry.periods)
            hk = kwant2hk(syst, params)
        except:
            return syst.hamiltonian_submatrix(params=params)
    elif type(syst) == dict:
        dim = len(next(iter(syst)))
        if dim == 0:
            return syst[next(iter(syst))]
        else:
            hk = dict2hk(syst)

    ks = 2 * np.pi * np.linspace(0, 1, nk, endpoint=False)

    k_pts = np.tile(ks, dim).reshape(dim, nk)

    ham = []
    for k in product(*k_pts):
        ham.append(hk(k))
    ham = np.array(ham)
    shape = (*[nk] * dim, ham.shape[-1], ham.shape[-1])

    if return_ks:
        return ham.reshape(*shape), ks
    else:
        return ham.reshape(*shape)


def build_interacting_syst(syst, lattice, func_onsite, func_hop, max_neighbor=1):
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
    syst_V : `kwant.Builder`
        Dummy `kwant.Builder` to compute interaction matrix.
    """
    syst_V = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    syst_V[syst.sites()] = func_onsite
    for neighbors in range(max_neighbor):
        syst_V[lattice.neighbors(neighbors + 1)] = func_hop
    return syst_V


def generate_guess(nk, syst_V, scale=0.1):
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
    ndof = syst_V[next(iter(syst_V))].shape[-1]
    guess = {}
    for vector in syst_V.keys():
        amplitude = np.random.rand(ndof, ndof)
        phase = 2 * np.pi * np.random.rand(ndof, ndof)
        rand_hermitian = amplitude * np.exp(1j * phase)
        rand_hermitian += rand_hermitian.T.conj()
        rand_hermitian /= 2
        guess[vector] = rand_hermitian

    return kgrid_hamiltonian(nk, guess) * scale


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


def hk2tb_model(hk, tb_model, int_model, ks):
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
    hopping_vecs = np.unique(np.array([*tb_model.keys(), *int_model.keys()]), axis=0)
    ndim = len(hk.shape) - 2
    dk = np.diff(ks)[0]
    nk = len(ks)
    k_pts = np.tile(ks, ndim).reshape(ndim, nk)
    k_grid = np.array(np.meshgrid(*k_pts))
    k_grid = k_grid.reshape(k_grid.shape[0], np.prod(k_grid.shape[1:]))
    hk = hk.reshape(np.prod(hk.shape[:ndim]), *hk.shape[-2:])

    hopps = (
        np.einsum(
            "ij,jkl->ikl",
            np.exp(1j * np.einsum("ij,jk->ik", hopping_vecs, k_grid)),
            hk,
        )
        * (dk / (2 * np.pi)) ** ndim
    )

    tb_model = {}
    for i, vector in enumerate(hopping_vecs):
        tb_model[tuple(vector)] = hopps[i]

    return tb_model


def hk_densegrid(hk, ks, nk_dense):
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
    tb_model = hk2hop(hk, ks)
    return kgrid_hamiltonian(nk_dense, tb_model)


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
