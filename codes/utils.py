# %%
import numpy as np
import kwant
from itertools import product


# %%
def get_fermi_energy(vals, filling):
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
    V = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    V[syst.sites()] = func_onsite
    for neighbors in range(max_neighbor):
        V[lattice.neighbors(neighbors + 1)] = func_hop
    wrapped_V = kwant.wraparound.wraparound(V).finalized()
    return syst2hamiltonian(ks=ks, syst=wrapped_V, params=params)


def assign_kdependence(
    ks, hopping_vecs, hopping_matrices
):
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
    keep = None
    deltas = []
    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        b_dom = builder.symmetry.which(b)
        # Throw away part that is in the remaining translation direction, so we get
        # an element of 'sym' which is being wrapped
        b_dom = np.array([t for i, t in enumerate(b_dom) if i != keep])
        deltas.append(b_dom)
    return np.asarray(deltas)


def generate_scf_syst(max_neighbor, syst, lattice):
    subs = np.array(lattice.sublattices)

    def scf_onsite(site, mat):
        idx = np.where(subs == site.family)[0][0]
        return mat[idx, 0] + 1j * mat[idx, 1]

    scf = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    scf[syst.sites()] = scf_onsite
    for neighbor in range(max_neighbor):

        def scf_hopping(site1, site2, mat):
            return (
                mat[len(lattice.sublattices) + neighbor, 0]
                + 1j * mat[len(lattice.sublattices) + neighbor, 1]
            )

        scf[lattice.neighbors(neighbor + 1)] = scf_hopping
    deltas = extract_hopping_vectors(scf)
    wrapped_scf = kwant.wraparound.wraparound(scf).finalized()
    return wrapped_scf, deltas


def hk2hop(hk, deltas, ks):
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
            np.exp(1j * np.einsum("ij,jk->ik", deltas, k_grid)),
            hk,
        )
        * (dk / (2 * np.pi)) ** ndim
    )

    return hopps


def hk_densegrid(hk, ks, ks_dense, hopping_vecs):
    """function is basically tiny so maybe don't separapetly create it"""
    hops = hk2hop(hk, hopping_vecs, ks)
    return assign_kdependence(ks_dense, hopping_vecs, hops)


def hk2syst(deltas, hk, ks, dk, max_neighbor, norbs, lattice):
    hopps = hk2hop(hk, deltas, ks, dk)
    bulk_scf = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    for i, delta in enumerate(deltas):
        for j, sublattice1 in enumerate(lattice.sublattices):
            for k, sublattice2 in enumerate(lattice.sublattices):
                if np.allclose(delta, [0, 0]):
                    bulk_scf[sublattice1.shape((lambda pos: True), (0, 0))] = hopps[
                        i, j * norbs : (j + 1) * norbs, j * norbs : (j + 1) * norbs
                    ]
                    if k != j:
                        hopping = (delta, sublattice1, sublattice2)
                        bulk_scf[kwant.builder.HoppingKind(*hopping)] = hopps[
                            i, j * norbs : (j + 1) * norbs, k * norbs : (k + 1) * norbs
                        ]
                else:
                    for k, sublattice2 in enumerate(lattice.sublattices):
                        hopping = (delta, sublattice1, sublattice2)
                        bulk_scf[kwant.builder.HoppingKind(*hopping)] = hopps[
                            i, j * norbs : (j + 1) * norbs, k * norbs : (k + 1) * norbs
                        ]
    wrapped_scf_syst = kwant.wraparound.wraparound(bulk_scf).finalized()
    return wrapped_scf_syst


def calc_gap(vals, E_F):
    emax = np.max(vals[vals <= E_F])
    emin = np.min(vals[vals > E_F])
    return np.abs(emin - emax)
