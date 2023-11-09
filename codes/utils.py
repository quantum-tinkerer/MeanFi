import numpy as np
import kwant
from itertools import product
from scipy.sparse import coo_array
import inspect
from copy import copy


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


def builder2tb_model(builder, params={}, return_data=False):
    """
    Constructs a tight-binding model dictionary from a `kwant.Builder`.

    Parameters:
    -----------
    builder : `kwant.Builder`
        Either builder for non-interacting system or interacting Hamiltonian.
    params : dict
        Dictionary of parameters to evaluate the Hamiltonian.
    return_data : bool
        Returns dictionary with sites and number of orbitals per site.

    Returns:
    --------
    tb_model : dict
        Tight-binding model of non-interacting systems.
    data : dict
        Data with sites and number of orbitals. Only if `return_data=True`.
    """
    builder = copy(builder)
    # Extract information from builder
    dims = len(builder.symmetry.periods)
    onsite_idx = tuple([0] * dims)
    tb_model = {}
    sites_list = [*builder.sites()]
    norbs_list = [site[0].norbs for site in builder.sites()]
    positions_list = [site[0].pos for site in builder.sites()]
    norbs_tot = sum(norbs_list)
    # Extract onsite and hopping matrices.
    # Based on `kwant.wraparound.wraparound`
    # Onsite matrices
    for site, val in builder.site_value_pairs():
        site = builder.symmetry.to_fd(site)
        atom = sites_list.index(site)
        row = np.sum(norbs_list[:atom]) + range(norbs_list[atom])
        col = copy(row)
        row, col = np.array([*product(row, col)]).T
        try:
            for arg in inspect.getfullargspec(val).args:
                _params = {}
                if arg in params:
                    _params[arg] = params[arg]
            val = val(site, **_params)
            data = val.flatten()
        except:
            data = val.flatten()
        if onsite_idx in tb_model:
            tb_model[onsite_idx] += coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
        else:
            tb_model[onsite_idx] = coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
    # Hopping matrices
    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        b_dom = builder.symmetry.which(b)
        b_fd = builder.symmetry.to_fd(b)
        atoms = np.array([sites_list.index(a), sites_list.index(b_fd)])
        row, col = [
            np.sum(norbs_list[: atoms[0]]) + range(norbs_list[atoms[0]]),
            np.sum(norbs_list[: atoms[1]]) + range(norbs_list[atoms[1]]),
        ]
        row, col = np.array([*product(row, col)]).T
        try:
            for arg in inspect.getfullargspec(val).args:
                _params = {}
                if arg in params:
                    _params[arg] = params[arg]
            val = val(a, b, **_params)
            data = val.flatten()
        except:
            data = val.flatten()
        if tuple(b_dom) in tb_model:
            tb_model[tuple(b_dom)] += coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
            if np.linalg.norm(b_dom) == 0:
                tb_model[tuple(b_dom)] += (
                    coo_array((data, (row, col)), shape=(norbs_tot, norbs_tot))
                    .toarray()
                    .T.conj()
                )
        else:
            tb_model[tuple(b_dom)] = coo_array(
                (data, (row, col)), shape=(norbs_tot, norbs_tot)
            ).toarray()
            if np.linalg.norm(b_dom) == 0:
                tb_model[tuple(b_dom)] += (
                    coo_array((data, (row, col)), shape=(norbs_tot, norbs_tot))
                    .toarray()
                    .T.conj()
                )

    if return_data:
        data = {}
        data["norbs"] = norbs_list
        data["positions"] = positions_list
        return tb_model, data
    else:
        return tb_model


def model2hk(tb_model):
    """
    Build Bloch Hamiltonian.

    Paramters:
    ----------
    nk : int
        Number of k-points along each direction.
    tb_model : dictionary
        Must have the following structure:
            - Keys are tuples for each hopping vector (in units of lattice vectors).
            - Values are hopping matrices.
    return_ks : bool
        Return k-points.

    Returns:
    --------
    ham : nd.array
        Hamiltonian evaluated on a k-point grid from k-points
        along each direction evaluated from zero to 2*pi.
        The indices are ordered as [k_1, ... , k_n, i, j], where
        `k_m` corresponding to the k-point element along each
        direction and `i` and `j` are the internal degrees of freedom.
    ks : 1D-array
        List of k-points over all directions. Only returned if `return_ks=True`.

    Returns:
    --------
    bloch_ham : function
        Evaluates the Hamiltonian at a given k-point.
    """
    assert (
        len(next(iter(tb_model))) > 0
    ), "Zero-dimensional system. The Hamiltonian is simply tb_model[()]."

    def bloch_ham(k):
        ham = 0
        for vector in tb_model.keys():
            ham += tb_model[vector] * np.exp(
                1j * np.dot(k, np.array(vector, dtype=float))
            )
        return ham

    return bloch_ham


def kgrid_hamiltonian(nk, hk, dim, return_ks=False):
    """
    Evaluates Hamiltonian on a k-point grid.

    Paramters:
    ----------
    nk : int
        Number of k-points along each direction.
    hk : function
        Calculates the Hamiltonian at a given k-point.
    return_ks : bool
        If `True`, returns k-points.

    Returns:
    --------
    ham : nd.array
        Hamiltonian evaluated on a k-point grid from k-points
        along each direction evaluated from zero to 2*pi.
        The indices are ordered as [k_1, ... , k_n, i, j], where
        `k_m` corresponding to the k-point element along each
        direction and `i` and `j` are the internal degrees of freedom.
    ks : 1D-array
        List of k-points over all directions. Only returned if `return_ks=True`.
    """
    ks = 2 * np.pi * np.linspace(0, 1, nk, endpoint=False)

    k_pts = np.tile(ks, dim).reshape(dim, nk)

    ham = []
    for k in product(*k_pts):
        ham.append(hk(k))
    ham = np.array(ham)
    assert np.allclose(
        ham, np.transpose(ham, (0, 2, 1)).conj()
    ), "Tight-binding provided is non-Hermitian. Not supported yet"
    shape = (*[nk] * dim, ham.shape[-1], ham.shape[-1])
    if return_ks:
        return ham.reshape(*shape), ks
    else:
        return ham.reshape(*shape)


def build_interacting_syst(builder, lattice, func_onsite, func_hop, max_neighbor=1):
    """
    Construct an auxiliary `kwant` system to build Hamiltonian matrix.

    Parameters:
    -----------
    builder : `kwant.Builder`
        Non-interacting `kwant` system.
    lattice : `kwant.lattice`
        System lattice.
    func_onsite : function
        Onsite function.
    func_hop : function
        Hopping function.
    max_neighbor : int
        Maximal nearest-neighbor order.

    Returns:
    --------
    int_builder : `kwant.Builder`
        Dummy `kwant.Builder` to compute interaction matrix.
    """
    int_builder = kwant.Builder(kwant.TranslationalSymmetry(*builder.symmetry.periods))
    int_builder[builder.sites()] = func_onsite
    for neighbors in range(max_neighbor):
        int_builder[lattice.neighbors(neighbors + 1)] = func_hop
    return int_builder


def generate_guess(vectors, ndof, scale=0.1):
    """
    nk : int
        Number of k-points along each direction.
    scale : float
        The scale of the guess. Maximum absolute value of each element of the guess.

    Returns:
    --------
    guess : nd-array
        Guess evaluated on a k-point grid.
    """
    guess = {}
    for vector in vectors:
        amplitude = scale * np.random.rand(ndof, ndof)
        phase = 2 * np.pi * np.random.rand(ndof, ndof)
        rand_hermitian = amplitude * np.exp(1j * phase)
        if np.linalg.norm(np.array(vector)):
            rand_hermitian += rand_hermitian.T.conj()
            rand_hermitian /= 2
        guess[vector] = rand_hermitian

    return guess


def generate_vectors(cutoff, dim):
    return [*product(*([[*range(-cutoff, cutoff + 1)]] * dim))]


def hk2tb_model(hk, hopping_vecs, ks=None):
    """
    Extract hopping matrices from Bloch Hamiltonian.

    Parameters:
    -----------
    hk : nd-array
        Bloch Hamiltonian matrix hk[k_x, ..., k_n, i, j]
    tb_model : dict
        Tight-binding model of non-interacting systems.
    int_model : dict
        Tight-binding model for interacting Hamiltonian.
    ks : 1D-array
        Set of k-points. Repeated for all directions. If the system is finite, `ks=None`.

    Returns:
    --------
    scf_model : dict
        TIght-binding model of Hartree-Fock solution.
    """
    if ks is not None:
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
    else:
        return {(): hk}


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
